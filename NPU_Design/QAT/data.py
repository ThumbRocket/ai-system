import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from config import batch_size, dataset


if dataset == dsets.MNIST:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
elif dataset == dsets.FashionMNIST:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

dset_train = dataset(
    root="./",  # 다운로드 경로 지정
    train=True,  # True를 지정하면 훈련 데이터로 다운로드
    transform=transform,
    download=True,
)

dset_val = dataset(
    root="./",  # 다운로드 경로 지정
    train=False,  # False를 지정하면 테스트 데이터로 다운로드
    transform=transform,
    download=True,
)


train_loader = torch.utils.data.DataLoader(
    dataset=dset_train, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    dataset=dset_val, batch_size=batch_size, shuffle=True
)
