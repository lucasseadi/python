import importlib
import json
import logging
import numpy as np
import pathlib
import random
import time
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from dataloader import get_loader
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

logging.basicConfig(format="[%(asctime)s %(name)s %(levelname)s] - %(message)s", datefmt="%Y/%m/%d %H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0

# model params
model_name = 'resNet'
model_block_type = 'basic'
model_depth = 20
model_base_channels = 16
model_input_shape = (1, 3, 32, 32)
model_n_classes = 10

# run params
run_outdir = ''
run_seed = 0
run_num_workers = 4
run_device = 'cuda'
run_tensorboard = True

# train params
train_optimizer = 'sgd'
train_epochs = 160
train_batch_size = 128
train_base_lr = 0.1
train_weight_decay = 1e-4
train_momentum = 0.9
train_nesterov = True

# scheduler params
scheduler_name = 'multistep'
scheduler_multistep_milestones = [80, 120]
scheduler_multistep_lr_decay = 0.1
scheduler_cosine_lr_min = 0


def load_model():
    module = importlib.import_module(model_name)
    Network = getattr(module, "Network")

    config = {"input_shape": model_input_shape, "n_classes": model_n_classes, "base_channels": model_base_channels,
              "block_type": model_block_type, "depth": model_depth}

    return Network(config)


def get_optimizer(model):
    if train_optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_base_lr, momentum=train_momentum,
                                    weight_decay=train_weight_decay, nesterov=train_nesterov)
    else:
        raise ValueError()
    return optimizer


def get_scheduler(optimizer):
    if scheduler_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_multistep_milestones,
                                                         gamma=scheduler_multistep_lr_decay)
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, scheduler_cosine_lr_min)
    else:
        raise ValueError()
    return scheduler


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(epoch, model, optimizer, scheduler, criterion, train_loader, writer):
    global global_step
    global run_device

    logger.info(f"Train {epoch}")

    model.train()
    device = torch.device(run_device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if run_tensorboard and step == 0:
            image = torchvision.utils.make_grid(data, normalize=True, scale_each=True)
            writer.add_image("Train/Image", image, epoch)

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        acc_meter.update(accuracy, num)

        if run_tensorboard:
            writer.add_scalar("Train/RunningLoss", loss_, global_step)
            writer.add_scalar("Train/RunningAccuracy", accuracy, global_step)

        if step % 100 == 0:
            logger.info(f"Epoch {epoch} Step {step}/{len(train_loader)} "
                        f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                        f"Accuracy {acc_meter.val:.4f} ({acc_meter.avg:.4f})")

    elapsed = time.time() - start
    logger.info(f"Elapsed {elapsed:.2f}")

    if run_tensorboard:
        writer.add_scalar("Train/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Train/Accuracy", acc_meter.avg, epoch)
        writer.add_scalar("Train/Time", elapsed, epoch)
        writer.add_scalar("Train/lr", scheduler.get_lr()[0], epoch)

    train_log = OrderedDict({"epoch": epoch, "train": OrderedDict({"loss": loss_meter.avg, "accuracy": acc_meter.avg,
                                                                   "time": elapsed})})
    return train_log


def test(epoch, model, criterion, test_loader, writer):
    logger.info(f"Test {epoch}")

    model.eval()
    device = torch.device(run_device)

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            if run_tensorboard and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(data, normalize=True, scale_each=True)
                writer.add_image("Test/Image", image, epoch)

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    logger.info(f"Epoch {epoch} Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}")

    elapsed = time.time() - start
    logger.info(f"Elapsed {elapsed:.2f}")

    if run_tensorboard:
        if epoch > 0:
            writer.add_scalar("Test/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Test/Accuracy", accuracy, epoch)
        writer.add_scalar("Test/Time", elapsed, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    test_log = OrderedDict({"epoch": epoch, "test": OrderedDict({"loss": loss_meter.avg, "accuracy": accuracy,
                                                                 "time": elapsed})})
    return test_log


def main():
    global run_device

    if not torch.cuda.is_available():
        run_device = "cpu"
    logger.info(json.dumps((train_batch_size, run_num_workers, run_device), indent=2))

    # set random seed
    seed = run_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # create output directory
    outdir = pathlib.Path(run_outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # TensorBoard SummaryWriter
    writer = SummaryWriter(outdir.as_posix()) if run_tensorboard else None

    # data loaders
    train_loader, test_loader = get_loader(train_batch_size, run_num_workers)

    # model
    model = load_model()
    model.to(torch.device(run_device))

    criterion = nn.CrossEntropyLoss(reduction="mean")

    # optimizer
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # run test before start training
    test(0, model, criterion, test_loader, writer)

    epoch_logs = []
    for epoch in range(train_epochs):
        epoch += 1
        scheduler.step()

        train_log = train(epoch, model, optimizer, scheduler, criterion, train_loader, writer)
        test_log = test(epoch, model, criterion, test_loader, writer)

        epoch_log = train_log.copy()
        epoch_log.update(test_log)
        epoch_logs.append(epoch_log)
        with open(outdir / "log.json", "w") as fout:
            json.dump(epoch_logs, fout, indent=2)

        state = OrderedDict([("state_dict", model.state_dict()),
                             ("optimizer", optimizer.state_dict()), ("epoch", epoch),
                             ("accuracy", test_log["test"]["accuracy"])])
        model_path = outdir / "model_state.pth"
        torch.save(state, model_path)


if __name__ == "__main__":
    main()
