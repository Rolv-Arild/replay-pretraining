import argparse
import random

import numpy as np
import torch

import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from idm_dataset import IDMDataset
from replay_pretraining.utils.util import ControlsPredictorDot


class IDMNet(nn.Module):
    def __init__(self, ff_dim, hidden_layers, dropout_rate):
        super().__init__()
        # self.conv = nn.Conv1d(45, conv_channels, kernel_size=(41,), stride=(1,))
        self.lin0 = nn.Linear(77 * 16, ff_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(ff_dim, ff_dim) for _ in range(hidden_layers)])
        self.dropout = nn.Dropout(dropout_rate)

        self.action_out = ControlsPredictorDot(ff_dim, ff_dim, ff_dim // 16, 2)
        self.on_ground_out = nn.Linear(ff_dim, 2)
        self.has_jump_out = nn.Linear(ff_dim, 2)
        self.has_flip_out = nn.Linear(ff_dim, 2)

    def forward(self, x: torch.Tensor, action_options: torch.Tensor):
        # x = self.conv(x.swapaxes(1, 2))
        x = self.lin0(torch.reshape(x, (x.shape[0], 77 * 16)))
        x = F.gelu(self.dropout(x))
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.gelu(self.dropout(x))
        return (
            self.action_out(x, action_options),
            self.on_ground_out(x),
            self.has_jump_out(x),
            self.has_flip_out(x)
        )


def collate(batch):
    x = [[] for _ in range(2)]
    y = [[] for _ in range(4)]
    for b in batch:
        for i in range(len(x)):
            x[i].append(b[0][i])
        for i in range(len(y)):
            y[i].append(b[1][i])
    return (tuple(torch.from_numpy(np.stack(x[i])).float() for i in range(len(x))),
            tuple(torch.from_numpy(np.stack(y[i])).long() for i in range(len(y))))


def train_idm(dataset_location, mutate_train_actions, ff_dim, hidden_layers, dropout_rate, lr, batch_size, epochs):
    assert torch.cuda.is_available()
    output_names = ["action", "on_ground", "has_jump", "has_flip"]
    train_dataset = IDMDataset(dataset_location, "train", mutate_action=mutate_train_actions)
    val_dataset = IDMDataset(dataset_location, "validation", limit=1, shuffle=False, corrupt=False)

    model = IDMNet(ff_dim, hidden_layers, dropout_rate)
    print(model)
    model.cuda()
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss(reduce=False)
    scheduler = None  # LambdaLR(optimizer, lr_lambda=lambda e: 1 / (0.2 * e + 1))

    logger = wandb.init(group="idm", project="replay-model", entity="rolv-arild",
                        config=dict(ff_dim=ff_dim, hidden_layers=hidden_layers,
                                    dropout_rate=dropout_rate, lr=lr, batch_size=batch_size,
                                    mutate_train_actions=mutate_train_actions,
                                    optimizer=type(optimizer).__name__, lr_schedule=scheduler is not None))

    min_loss = float("inf")

    steps_per_hour = 108000
    assert steps_per_hour % batch_size == 0
    train_log_rate = 1
    val_log_rate = 24

    n = 0

    train_losses = {k: 0 for k in output_names}
    train_inferences = 0
    train_correct = {k: 0 for k in output_names}
    train_samples = 0
    l2_norm = 0
    model_mag = 0
    update_size = 0
    for epoch in range(epochs):
        model.train()
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate, drop_last=True)
        for x_train, y_train in train_loader:
            assert len(x_train[0]) == len(y_train[0]) == batch_size, f"{len(x_train)}, {len(y_train)}"
            total_samples = batch_size * n

            if total_samples % (val_log_rate * steps_per_hour) == 0:
                print("Validating...")
                with torch.no_grad():
                    model.eval()
                    val_loader = DataLoader(val_dataset, 9000, collate_fn=collate)
                    val_losses = {k: 0 for k in output_names}
                    val_inferences = 0
                    val_correct = {k: 0 for k in output_names}
                    val_samples = 0
                    for x_val, y_val in val_loader:
                        y_hat = model(x_val[0].cuda(), x_val[1].cuda())
                        for i, name in enumerate(output_names):
                            val_losses[name] += loss_fn(y_hat[i], y_val[i].cuda()).sum().item()
                            val_correct[name] += (y_hat[i].cpu().argmax(axis=-1) == y_val[i]).sum().item()
                        val_samples += len(x_val[0])
                        val_inferences += 1
                    val_losses = {k: v / val_samples for k, v in val_losses.items()}
                    accuracies = {k: v / val_samples for k, v in val_correct.items()}
                    loss = sum(val_losses.values())
                    logger.log({"validation/loss": loss}, commit=False)
                    logger.log({f"validation/{k}_loss": v for k, v in val_losses.items()}, commit=False)
                    logger.log({f"validation/{k}_accuracy": v for k, v in accuracies.items()}, commit=False)
                    print(f"Day {total_samples // (val_log_rate * steps_per_hour)}:", loss, val_losses)
                    if loss < min_loss:
                        mdl_jit = torch.jit.trace(model.cpu(), (torch.zeros(10, 77, 16), torch.zeros(10, 9, 8)))
                        torch.jit.save(mdl_jit, f"models/idm-model-{logger.name}.pt")
                        model.cuda()
                        print(f"Model saved at day {total_samples // (val_log_rate * steps_per_hour)} "
                              f"with total validation loss {loss}")
                        min_loss = loss
                model.train()

            y_hat = model(x_train[0].cuda(), x_train[1].cuda())
            loss = 0
            for i, name in enumerate(output_names):
                ls = loss_fn(y_hat[i], y_train[i].cuda())
                l = ls.mean()
                loss += l
                train_losses[name] += l.item()
                train_correct[name] += (y_hat[i].cpu().argmax(axis=-1) == y_train[i]).sum().item()
            train_inferences += 1
            train_samples += len(x_train[0])

            loss.backward()
            with torch.no_grad():
                gradient = [param.grad.clone() for param in model.parameters() if param.grad is not None]
                l2_norm += torch.sqrt(sum([(g ** 2).sum() for g in gradient])).item()
            # Store weights of model, so we can calculate the size of the update
            old_weights = [param.data.clone() for param in model.parameters()]

            optimizer.step()

            # Calculate the size of the update
            update_size += torch.sqrt(
                sum([((param.data - old) ** 2).sum() for param, old in zip(model.parameters(), old_weights)])).item()
            model_mag += torch.sqrt(sum([(param.data ** 2).sum() for param in model.parameters()])).item()

            model.zero_grad(True)
            n += 1

            if total_samples % (train_log_rate * steps_per_hour) == 0:
                train_losses = {k: v / train_inferences for k, v in train_losses.items()}
                # if train_losses["action"] > 1:
                #     print("Hei")
                #     print(y_hat[0].min().item())

                logger.log({"epoch": epoch}, commit=False)
                logger.log({"train/samples": total_samples}, commit=False)
                logger.log({"train/loss": sum(train_losses.values())}, commit=False)
                logger.log({f"train/{k}_loss": v for k, v in train_losses.items()}, commit=False)
                logger.log({f"train/{k}_accuracy": v / train_samples for k, v in train_correct.items()}, commit=False)
                logger.log({"train/update_size": update_size / train_inferences}, commit=False)
                logger.log({"train/l2_norm": l2_norm / train_inferences}, commit=False)
                logger.log({"train/model_mag": model_mag / train_inferences}, commit=False)
                logger.log({"train/learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else lr})
                print(f"Hour {total_samples // steps_per_hour}:", sum(train_losses.values()))
                train_losses = {k: 0 for k in output_names}
                train_inferences = 0
                train_correct = {k: 0 for k in output_names}
                train_samples = 0
                l2_norm = 0
                model_mag = 0
                update_size = 0
                if scheduler is not None:
                    scheduler.step()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_location", type=str, required=True)
    parser.add_argument("--mutate_train_actions", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    seed_everything(args.seed)
    train_idm(
        dataset_location=args.dataset_location,
        mutate_train_actions=args.mutate_train_actions,
        ff_dim=args.ff_dim,
        hidden_layers=args.hidden_layers,
        dropout_rate=args.dropout_rate,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
