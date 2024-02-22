import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb

from replay_pretraining.bcm.bc_dataset import BCDataset, CombinedDataset
from replay_pretraining.bcm.bc_net import BCNet


def collate(batch):
    x, y = zip(*batch)
    # for b in batch:
    #     x.append(b[0])
    #     y.append(b[1])
    # x = [np.pad(xs, pad_width=(0, 80 + 231 - xs.shape[0])) for xs in x]
    return (torch.from_numpy(np.stack(x)).float(),
            torch.from_numpy(np.stack(y)).long())


def train_bc():
    assert torch.cuda.is_available()
    # train_dataset = BCDataset(r"E:\rokutleg\ssl-dataset\ranked-doubles", "train")
    # val_dataset = BCDataset(r"E:\rokutleg\ssl-dataset\ranked-doubles", "validation", limit=10)

    train_dataset = CombinedDataset(
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-duels", "train"),
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-doubles", "train"),
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-standard", "train")
    )

    val_dataset = CombinedDataset(
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-duels", "validation", limit=10),
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-doubles", "validation", limit=10),
        BCDataset(r"E:\rokutleg\ssl-dataset\ranked-standard", "validation", limit=10)
    )

    ff_dim = 1024
    hidden_layers = 6
    dropout_rate = 0.
    lr = 5e-5
    batch_size = 300

    model = BCNet(231 + 0, ff_dim, hidden_layers, dropout_rate)
    print(model)
    j_inp = torch.normal(0, 1, (10, 231 + 0))
    j_inp[1, -(2 * 31 + 1)] = 0
    j_inp[2, -(4 * 31 + 1)] = 0
    model = torch.jit.trace(model, (j_inp,))  # , check_inputs=[(torch.cat((j_inp, j_inp), dim=0),)]).cuda()
    test = model(torch.cat((j_inp, j_inp), dim=0))
    test = model.cuda()(torch.cat((j_inp, j_inp), dim=0).cuda())
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1 / (0.25 * e + 1))

    logger = wandb.init(group="behavioral-cloning", project="replay-model", entity="rolv-arild",
                        config=dict(ff_dim=ff_dim, dropout_rate=dropout_rate, lr=lr, batch_size=batch_size,
                                    optimizer=type(optimizer).__name__, lr_schedule=scheduler is not None))

    min_loss = float("inf")

    steps_per_hour = 108000
    assert steps_per_hour % batch_size == 0
    train_log_rate = 1
    val_log_rate = 7 * 24

    n = 0
    tot_loss = k = 0
    for epoch in range(100):
        model.train()
        train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate, drop_last=True)
        for x_train, y_train in train_loader:
            total_samples = batch_size * n

            if total_samples % (val_log_rate * steps_per_hour) == 0:
                print("Validating...")
                with torch.no_grad():
                    model.eval()
                    val_loader = DataLoader(val_dataset, 9000, collate_fn=collate)
                    loss = m = 0
                    correct = total = 0
                    for x_val, y_val in val_loader:
                        y_hat = model(x_val.cuda())
                        loss += loss_fn(y_hat, y_val.cuda()).item()
                        correct += (y_hat.cpu().argmax(axis=-1) == y_val).sum().item()
                        total += len(y_val)
                        m += 1
                    loss /= m
                    logger.log({"validation/loss": loss}, commit=False)
                    logger.log({"validation/accuracy": correct / total}, commit=False)
                    print(f"Day {total_samples // (val_log_rate * steps_per_hour)}:", loss)
                    if loss < min_loss:
                        torch.jit.save(torch.jit.trace(model, j_inp), f"bc-model-{logger.name}.pt")
                        print(f"Model saved at day {total_samples // (val_log_rate * steps_per_hour)} "
                              f"with total validation loss {loss}")
                        min_loss = loss
                model.train()

            y_hat = model(x_train.cuda())
            loss = loss_fn(y_hat, y_train.cuda())
            tot_loss += loss.item()
            k += 1
            if total_samples % (train_log_rate * steps_per_hour) == 0:
                logger.log({"epoch": epoch}, commit=False)
                logger.log({"train/samples": total_samples}, commit=False)
                logger.log({"train/loss": tot_loss / k}, commit=False)
                logger.log({"train/learning_rate": scheduler.get_last_lr()[0]})
                print(f"Hour {total_samples // steps_per_hour}:", tot_loss / k)
                tot_loss = k = 0

            loss.backward()
            optimizer.step()
            model.zero_grad(True)
            n += 1
        scheduler.step()


def test_bc():
    test_dataset = BCDataset(r"D:\rokutleg\ssl-dataset-fixed", "test")
    model = torch.jit.load("bc-model.pt").cuda()

    with open("../../bc-results.csv", "w") as f:
        f.write("action_pred,action_true\n")
        with torch.no_grad():
            model.eval()
            test_loader = DataLoader(test_dataset, 12, collate_fn=collate)
            for x, y_true in test_loader:
                y_hat = model(x.cuda())
                for j in range(len(x)):
                    f.write(f"{y_hat[j].argmax(axis=-1).item()},{y_true[j]}\n")


if __name__ == '__main__':
    # import cProfile, pstats, io
    # from pstats import SortKey
    #
    # pr = cProfile.Profile()
    # pr.enable()

    # make_bc_dataset(r"E:\rokutleg\parsed\2021-ssl-replays\ranked-doubles",
    #                 r"E:\rokutleg\ssl-dataset\ranked-doubles")

    # make_bc_dataset(r"E:\rokutleg\parsed\2021-ssl-replays\ranked-duels",
    #                 r"E:\rokutleg\ssl-dataset\ranked-duels")

    # make_bc_dataset(r"E:\rokutleg\parsed\2021-ssl-replays\ranked-standard",
    #                 r"E:\rokutleg\ssl-dataset\ranked-standard")

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    #
    # ps.dump_stats("profile_results.pstat")

    # time.sleep(3 * 60 * 60)
    train_bc()
    # test_bc()
