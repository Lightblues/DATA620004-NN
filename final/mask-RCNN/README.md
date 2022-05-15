- detectron2
- [model](https://dl.fbaipublicfiles.com/detectron2/Cityscapes/mask_rcnn_R_50_FPN/142423278/model_final_af9cf5.pkl)
- [video](https://www.youtube.com/watch?v=3-DwOlaekow)


```sh
youtube-dl https://www.youtube.com/watch?v=3-DwOlaekow -f 22 -o video.mp4
ffmpeg -i video.mp4 -t 00:00:06 -c:v copy video-clip.mp4

python demo.py --config-file configs/Cityscapes/mask_rcnn_R_50_FPN.yaml \
    --video-input video-clip.mp4 --confidence-threshold 0.6 --output video-output.mkv \
    --opts MODEL.WEIGHTS model_final_af9cf5.pkl
```
