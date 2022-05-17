
- 采用包: <https://github.com/facebookresearch/detectron2>
- 模型: <https://github.com/Lightblues/NN-pj/releases/tag/v0.0.3>
- 模型初始化参数
    - [ImageNet](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl)
    - [mask rcnn](https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl)

### 训练

准备VOC数据, 放在 `DETECTRON2_DATASETS` 目录下.

```sh
export DETECTRON2_DATASETS=../datasets/
export CONFIG_PATH=./configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml

# train
# 2.0 随机初始化训练, 为了避免梯度爆炸降低lr
python -u ./train_net.py  \
 OUTPUT_DIR "./logs/v2.0" MODEL.WEIGHTS "" SOLVER.BASE_LR 0.0005 
# 2.1 用 ImageNet 预训练模型
python -u ./train_net.py  \
 OUTPUT_DIR "./logs/v2.1" MODEL.WEIGHTS "R-50.pkl" SOLVER.BASE_LR 0.005
# 2.2 使用coco训练的Mask R-CNN的backbone网络参数初始化Faster R-CNN的backbone网络
python -u ./train_net.py  \
 OUTPUT_DIR "./logs/v2.2" MODEL.WEIGHTS "model_final_f10217.pkl" SOLVER.BASE_LR 0.005

```

### 绘图

```sh
# 绘制 bbox
python get_boundingbox.py --config-file $CONFIG_PATH \
    --input ../sample-fig/cat.jpg ../sample-fig/furniture.jpg ../sample-fig/persons.jpg \
    --output ./output_boundingbox \
    --opts MODEL.WEIGHTS $MODEL_PATH MODEL.DEVICE cpu
```

### 实验设置

- 数据集: [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html).
    - 训练集: 2007trainval + 2012trainval
    - 测试集: 2007test
- 网络结构:
- batch size: 16
- iteration: about 80k
- learning rate: 0.005
- optimizer: SGD with momentum
- loss: total loss with 4 parts: cls, box reg, rpn cls, rpn reg.
- metric: mAP
