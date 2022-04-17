import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


from data_aug.cutout import Cutout
from data_aug.cutmix import CutMix
from data_aug.mixup import Mixup


transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
])

#transform_train.transforms.append(Cutout(1, length=16))

writer = SummaryWriter("original")
cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

for i in range(3):
    img , target = cifar100_training[i]
    writer.add_image("cifar100_training",img,i)

transform_train_cutout = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    Cutout(1, length=16)
])

cutout=torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_cutout)
for i in range(3):
    img,target=cutout[i]
    writer.add_image("cutout",img,i)


cutmix=CutMix(cifar100_training,num_class=100,beta=1.0,prob=1,num_mix=2)
for i in range(3):
    img,target=cutmix[i]
    writer.add_image("cutmix",img,i)

mixup=Mixup(cifar100_training,num_class=100,beta=1.0,prob=1)
for i in range(3):
    img,target=mixup[i]
    writer.add_image("mixup",img,i)

writer.close()