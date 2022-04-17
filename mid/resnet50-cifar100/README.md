# Resnet50-cifar100

resent for cifar100 with pytorch


## Train the model
training without data augmentation
```bash
python train.py -net resnet50 -gpu
```
training with data augmentation (cutmix as example, mixup,cutmix available)
```bash
python train.py -net resnet50 -gpu -data_aug cutmix
```

## Test the model
```bash
python test.py -net resnet50 -weights path/to/resnet50_weights
```

## Tensorboard
```bash
tensorboard --logdir='runs' --port=6006 --host='127.0.0.1'
```