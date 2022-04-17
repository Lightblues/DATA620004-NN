#CIFAR100 resnet-50
## 参数设置
网络结构：resnet-50
训练测试集划分:cifar100-train,cifar100-test
batch_size 128
learning_rate_initial 0.1
learning_rate_decay: 
multistep with milestone=[60,120,160] and $\gamma$=0.2
warmup from 0 to 0.1 in first epoch

optimzer momentum SGD with momentum=0.9 and decay_rate=5e-4
epoch  200
lossfunction  CrossEntropyLoss
评价指标 top1 error / top5 error


## 结果对比
model|data augmentation|top1_error|top5_error
---|:--:|:--:|:---:
resnet50|none|0.2147|0.0553 
resnet50 |mixup|0.2039|0.0566
resnet50 |cutout|0.2196|0.0589
resnet50 |cutmix|0.2131|0.0557
