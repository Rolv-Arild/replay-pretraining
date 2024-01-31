import copy
from typing import Callable, Union, Any, Dict, List

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from wandb.sdk.wandb_run import Run


def transform_batch(batch, fn):
    if isinstance(batch, tuple):
        return tuple(fn(t) for t in batch)
    elif isinstance(batch, list):
        return [fn(t) for t in batch]
    elif isinstance(batch, dict):
        return {k: fn(v) for k, v in batch.items()}
    else:
        return fn(batch)


class Metric:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, y_hat, y):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class Accuracy(Metric):
    def __init__(self, name: str):
        super().__init__(name)
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, y_hat, y):
        self.correct += (y_hat.argmax(dim=-1) == y).sum().item()
        self.total += y.shape[0]

    def compute(self):
        return self.correct / self.total


class MSE(Metric):
    def __init__(self, name: str):
        super().__init__(name)
        self.sum = 0
        self.total = 0

    def reset(self):
        self.sum = 0
        self.total = 0

    def update(self, y_hat, y):
        self.sum += ((y_hat - y) ** 2).sum().item()
        self.total += y.shape[0]

    def compute(self):
        return self.sum / self.total


class Trainer:
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            collate_fn: Callable,
            loss_fn: Callable[[Any, Any], torch.Tensor],
            device: Union[str, torch.device],
            epochs: int,
            batch_size: int,
            train_dataset: Dataset,
            val_dataset: Dataset,
            val_freq: int,  # Number of steps between validation runs
            train_metrics: List[Metric],
            val_metrics: List[Metric],
            logger: Run,
            callbacks: List[Callable],
    ):
        self.model = model
        self.optimizer = optimizer
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_freq = val_freq
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.logger = logger
        self.callbacks = callbacks

        self.model.to(self.device)

    def call_model(self, batch, metrics):
        x, y = batch
        x = transform_batch(x, lambda t: t.to(self.device))
        y = transform_batch(y, lambda t: t.to(self.device))
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        for metric in metrics:
            metric.update(y_hat, y)
        return loss

    def run_epoch(self, dataset, metrics):
        # Generalize to train/val
        loader = DataLoader(dataset, self.batch_size, collate_fn=self.collate_fn)
        for batch in loader:
            loss = self.call_model(batch, metrics)
            yield loss

    def run(self):
        batch_count = 0
        for epoch in range(self.epochs):
            iterator = self.run_epoch(self.train_dataset, self.train_metrics)
            while True:  # While instead of for to do validation first
                if batch_count % self.val_freq == 0:
                    losses = []
                    for loss in self.run_epoch(self.val_dataset, self.val_metrics):
                        losses.append(loss)
                    metrics = {
                        "validation/loss": sum(losses) / len(losses),
                        "epoch": epoch,
                        "total_steps": batch_count * self.batch_size
                    }
                    for metric in self.val_metrics:
                        metrics[metric.name] = metric.compute()
                        metric.reset()
                    self.logger.log(metrics)
                try:
                    loss, batch_metrics = next(iterator)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_count += 1
                except StopIteration:
                    break
