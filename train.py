#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import copy
import argparse
import yaml
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils, models
from tensorboardX import SummaryWriter

from models.resnet import ResNet18
from utils.data_loader import Pet_Dataset


parser = argparse.ArgumentParser()
parser.add_argument("--cfg",
                    type=str, dest="cfg",
                    default="configs/resnet18.yaml",
                    help="the path of training config yaml file."
                    )


def train_model(model, data_loader, criterion, optimizer, num_epochs=15):
    since = time.time()
    val_acc_history = []
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for i in range(num_epochs):
        print("Epoch: {}".format(i))
        print("_" * 10)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for images, labels in data_loader[phase]:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels)
            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_accuracy = running_corrects.double() / len(data_loader[phase].dataset)
            print("{} loss: {}, acc: {}.".format(phase, epoch_loss, epoch_accuracy))
            if phase == "train":
                writer_train.add_scalar('loss', epoch_loss, i)
                writer_train.add_scalar('acc', epoch_accuracy, i)
                writer_train.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
            else:
                writer_test.add_scalar('loss', epoch_loss, i)
                writer_test.add_scalar('acc', epoch_accuracy, i)
            if phase == "val":
                val_acc_history.append(epoch_accuracy)
                if epoch_accuracy > best_acc:
                    best_acc = epoch_accuracy
                    best_weights = copy.deepcopy(model.state_dict())

    print("Train completed in: {:.0f}min{:.0f}s.".format((time.time() - since) // 60, (time.time() - since) % 60))
    print("Best acuuracy: {:.4f}.".format(best_acc))
    model.load_state_dict(best_weights)
    return model, best_acc


def read_cfg(file):
    with open(file, 'r') as f:
        cfg = yaml.safe_load(f)
    # f = open(file)
    # cfg = yaml.safe_load(f)
    return edict(cfg)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    file = args.cfg
    cfg = read_cfg(file)
    print(cfg)
    image_path = cfg.train.dataset_path
    csv_path = cfg.train.csv_path
    batch_size = cfg.train.batch_size
    base_lr = cfg.solver.base_lr
    weight_decay = cfg.solver.weight_decay
    num_epochs = cfg.solver.num_epochs

    now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    writer_train = SummaryWriter('runs/{}/train'.format(now_time))
    writer_test = SummaryWriter('runs/{}/val'.format(now_time))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    train_dataset = Pet_Dataset(image_path, csv_path, phase="train", transform=data_transform["train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataset = Pet_Dataset(image_path, csv_path, phase="val", transform=data_transform["val"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    data_loader = {"train": train_loader, "val": val_loader}

    if cfg.model.type == "resnet18":
        model = ResNet18(label_number=37)
    if os.path.exists(cfg.model.weights):
        model.load_state_dict(torch.load(cfg.model.weights))
    # model = models.resnet18(num_classes=37)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    model, hist = train_model(model, data_loader, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)
    model_name = "{0}-{1:.4f}acc-{2}epochs-lr{3}-{4}weightdecay.pth".format(cfg.model.type, hist, num_epochs,
                                                                        base_lr, weight_decay)
    torch.save(model.state_dict(), os.path.join("runs/{}/".format(now_time), model_name))
    writer_train.close()
    writer_test.close()
