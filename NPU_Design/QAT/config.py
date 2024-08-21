import torchvision.datasets as dsets
import os

# manual
device = "gpu"
assert device in ["gpu", "cpu"]

from_scratch = False
learning_rate = 0.1
lr_base = 1e-2
pretrain_epoch = 40
batch_size = 64
quant_epoch = 20
warmup_ratio = 0.1

finetune_lr_ratio = 0.1

seed = 42

dataset = dsets.MNIST

Q = "NIPQ"
assert Q in ["STE", "NIPQ"]

Q_BITS = 2
Q_noise_type = "normal"
assert Q_noise_type in ["uniform", "normal", "normal_rounded"]

save_dir = "weight"

# determined
if not from_scratch:
    pretrain_epoch = 0
assert dataset in [dsets.MNIST, dsets.FashionMNIST]
dataset_str = "MNIST" if dataset == dsets.MNIST else "FashionMNIST"
num_classes = 10

pretrained = (
    "epoch40_0.9802_1.0.pth"
    if dataset == dsets.MNIST
    else "epoch40_0.8972_0.9954.pth"
)
pretrained = os.path.join(
    os.curdir, save_dir, dataset_str, "pretrain", pretrained
)

pretrain_warmup_epoch = int(pretrain_epoch * warmup_ratio)
quant_warmup_epoch = int(quant_epoch * warmup_ratio)
total_epoch = pretrain_epoch + quant_epoch

print("# Train schedule")
if from_scratch:
    print("- train from scratch")
else:
    print("- using pretrained model")
print(f"- start at {lr_base * learning_rate}")
print("- pretrain")
print(f"  - warmup up to epoch {pretrain_warmup_epoch} until {learning_rate}")
print(
    f"  - pretrain decay up to epoch {pretrain_epoch} until {lr_base * learning_rate}"
)
print("- QAT finetune")
print(
    f"  - warmup up to epoch {pretrain_epoch + quant_warmup_epoch} until {finetune_lr_ratio * learning_rate}"
)
print(
    f"  - pretrain decay up to epoch {total_epoch} until {lr_base * finetune_lr_ratio * learning_rate}"
)
print()
print("# Dataset")
print(f"- {dataset_str}")
print()
print("# Quant")
print(f"- {Q} method")
if Q == "NIPQ":
    print(f"- noise type : {Q_noise_type}")
print(f"- {Q_BITS} bit")
print()
print("# etc")
print(f"- seed = {seed}")

retrieve = input("Enter `Yes` to proceed: ")
if retrieve != "Yes":
    print(f"Retrieved {retrieve}, exit...")
    exit()


def lambda_lr(epoch):
    lr_scale = None
    base = lr_base

    if epoch < pretrain_epoch:
        # pretrain
        if epoch < pretrain_warmup_epoch:
            # warmup
            lr_scale = epoch * (1.0 - base) / pretrain_warmup_epoch + base
        else:
            # linear decay
            lr_scale = (
                -1
                * (epoch - pretrain_warmup_epoch)
                * (1.0 - base)
                / (pretrain_epoch - pretrain_warmup_epoch)
            ) + 1.0
    else:
        # quant
        if epoch - pretrain_epoch < quant_warmup_epoch:
            # warmup
            lr_scale = (epoch - pretrain_epoch) * (
                1.0 - base
            ) / quant_warmup_epoch + base
        else:
            # linear decay
            lr_scale = (
                -1
                * (epoch - quant_warmup_epoch - pretrain_epoch)
                * (1.0 - base)
                / (quant_epoch - quant_warmup_epoch)
                + 1.0
            )
        lr_scale *= finetune_lr_ratio
    assert 0.0 <= lr_scale <= 1.0
    return lr_scale
