#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.init
import torch.nn as nn
import os
from torchmetrics.functional.classification import multiclass_accuracy
from lightning.fabric import Fabric

from model import CNN
from config import (
    device,
    seed,
    save_dir,
    total_epoch,
    pretrain_epoch,
    pretrained,
    from_scratch,
    learning_rate,
    lambda_lr,
    dataset_str,
    num_classes,
    Q_BITS,
    Q,
)
from data import train_loader, val_loader


def eval_custom(model, num_classes, val_loader):
    model.eval()
    total = 0
    top_1_accurate = 0
    top_5_accurate = 0

    with torch.no_grad():
        for X, Y in val_loader:
            prediction = model(X)
            top_1_acc = multiclass_accuracy(
                prediction, Y, num_classes=num_classes, top_k=1
            )
            top_5_acc = multiclass_accuracy(
                prediction, Y, num_classes=num_classes, top_k=5
            )
            total += len(Y)
            top_1_accurate += int((top_1_acc * len(X)).item())
            top_5_accurate += int((top_5_acc * len(X)).item())

    top_1_acc = top_1_accurate / total
    top_5_acc = top_5_accurate / total
    return top_1_acc, top_5_acc


# ### iteration loop

# In[ ]:

fabric = None
if device == "cpu":
    fabric = Fabric(accelerator="cpu")
elif device == "gpu":
    fabric = Fabric(accelerator="gpu", devices=1)
else:
    raise ValueError(device)
fabric.launch()
fabric.seed_everything(seed)

train_loader = fabric.setup_dataloaders(train_loader)
val_loader = fabric.setup_dataloaders(val_loader)

model = CNN()
if not from_scratch:
    # load pretrained model
    model.load_state_dict(torch.load(pretrained))

criterion = (
    torch.nn.CrossEntropyLoss()
)  # 비용 함수에 소프트맥스 함수 포함되어져 있음.

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

model, optimizer = fabric.setup(model, optimizer)

total_batch = len(train_loader)
print("steps per batch : {}".format(total_batch))

best_acc = 1.0  # skip until QAT finetune
print(
    f"{'Epoch':8}, {'LR':8}, {'loss':8}, {'top_1':8}, {'top_5':8}, {'save':8}"
)

weight_root = os.path.join(os.curdir, save_dir)
if not os.path.exists(weight_root):
    os.mkdir(weight_root)
weight_per_dataset = os.path.join(weight_root, dataset_str)
if not os.path.exists(weight_per_dataset):
    os.mkdir(weight_per_dataset)

for epoch in range(total_epoch):
    if epoch + 1 > pretrain_epoch:
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.is_quant = True

    avg_loss = 0

    # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
    model.train()
    for X, Y in train_loader:
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        optimizer.zero_grad()
        hypothesis = model(X)
        loss = criterion(hypothesis, Y)
        fabric.backward(loss)
        optimizer.step()

        avg_loss += loss / total_batch
    top_1, top_5 = eval_custom(model, num_classes, val_loader)
    is_save = False
    if epoch % 10 == 9:
        is_save = True
    if (top_1 > best_acc) or (epoch + 1 > pretrain_epoch and best_acc == 1.0):
        best_acc = top_1
        is_save = True

    if is_save:
        stage = None
        if epoch + 1 <= pretrain_epoch:
            stage = "pretrain"
        else:
            stage = f"{Q}_{Q_BITS}bit"
        param_dir = os.path.join(weight_per_dataset, stage)
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)
        filename = (
            "_".join(
                [
                    f"epoch{epoch + 1}",
                    f"{top_1:.6}",
                    f"{top_5:.6}",
                ]
            )
            + ".pth"
        )
        torch.save(
            model.state_dict(),
            os.path.join(param_dir, filename),
        )

    print(
        "{:8}, {:8.4}, {:8.4}, {:8.4}, {:8.4}, {:8}".format(
            epoch + 1,
            scheduler.get_last_lr()[0],
            avg_loss,
            top_1,
            top_5,
            is_save,
        )
    )
    scheduler.step()
