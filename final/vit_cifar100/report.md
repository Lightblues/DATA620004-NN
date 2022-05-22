# CIFAR100 vit
## 参数设置
网络结构：Vision-Transformer 

网络参数:
parameter | value
:---:|:---:
patch_size | 8
dim| 512
depth | 6
heads| 6
mlp_dim|3072
dropout| 0.1
emb_dropout |0.1


训练测试集划分:cifar100-train,cifar100-test

超参数设置:
parameter | value
:---:|:---:
batch_size |128
learning_rate_initial| 1e-3
learning_rate_decay| CosineAnnealingLR with eta_min=1e-5
warmup |from 0 to 1e-3 in first epoch
optimzer | Adam with betas=(0.9,0.999)
weight_dacay | 5e-5
epoch|200
lossfunction | CrossEntropyLoss
评价指标 | top1 error / top5 error


## 结果对比
model| parameters_number |data augmentation|top1_error|top5_error
:---:|:--:|:--:|:--:|:---:
resnet50 |23,705,252 |cutout|0.2196|0.0589
resnet50|- |none|0.2147|0.0553
resnet50 |- |cutmix|0.2131|0.0557
resnet50 | -|mixup|0.2039|0.0566
vit | 23,790,180 | mixup | 0.4608 | 0.1994
