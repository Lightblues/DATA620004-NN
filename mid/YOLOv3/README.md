- 采用包: <https://github.com/ultralytics/yolov3>
- 训练日志见 <https://wandb.ai/easonshi/YOLOv3>
- 模型 <https://github.com/Lightblues/NN-pj/releases/tag/v0.0.1>

### 环境配置

按照 requirements.txt 配置环境

### 训练

```sh
# train
# 下载初始化参数
python train.py --img 640 --batch 16 --epochs 3 --data voc.yaml --weights yolov3.pt --cache --workers 4

export MODEL_PATH=runs/train/exp6/weights/best.pt
export DATA_PATH=../sample-fig/

# val
python val.py --weights $MODEL_PATH --data voc.yaml --img 640 --iou 0.65 --half
```

### 绘图

```sh
python detect.py --weights $MODEL_PATH --img 640 --conf 0.25 --source $DATA_PATH \
    --save-txt --save-crop --save-conf \
    --project output_boundingbox
```
