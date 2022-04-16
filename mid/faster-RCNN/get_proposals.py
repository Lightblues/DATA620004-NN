import argparse
from email.policy import default
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog
# from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, _create_text_labels

from detectron2.modeling import build_model, GeneralizedRCNN
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

WINDOW_NAME = "proposal_visualization"


""" 得到 proposals, 绘图. @220416
修改自 gemo/demo.py

python get_proposals.py --config-file ./configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml \
    --input ../sample-fig/test1.jpg ../sample-fig/cat.jpg ../sample-fig/furniture.jpg ../sample-fig/persons.jpg \
    --output ./output_proposals \
    --opts MODEL.WEIGHTS ./tools/output/model_final.pth MODEL.DEVICE cpu
"""

# from detectron2.modeling import build_model
# model = build_model(cfg)  # returns a torch.nn.Module

# from detectron2.checkpoint import DetectionCheckpointer
# DetectionCheckpointer(model).load(file_path_or_url)  # load a file, usually from cfg.MODEL.WEIGHTS

# checkpointer = DetectionCheckpointer(model, save_dir="output")
# checkpointer.save("model_999")  # save to output/model_999.pth


# images = ImageList.from_tensors(...)  # preprocessed input tensor
# model = build_model(cfg)
# model.eval()
# features = model.backbone(images.tensor)
# proposals, _ = model.proposal_generator(images, features)
# instances, _ = model.roi_heads(images, features, proposals)
# mask_features = [features[f] for f in model.roi_heads.in_features]
# mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "sample-fig/test1.jpg", # 在训练集合中
            "sample-fig/cat.jpg", "sample-fig/furniture.jpg", "sample-fig/Lenna.jpg"],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="output_proposals",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "./tools/output/model_final.pth", "MODEL.DEVICE", "cpu"],
        nargs=argparse.REMAINDER,
    )
    return parser

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
    
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

    def get_proposals(self, original_image):
        """ 得到 proposals
        1) 将图片经过 transformer后整理成 inputs
        2) 经过 backbone 和 proposal_generator;
        3) postprocess 变换为原图的大小
        """
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        # predictions = self.model([inputs])[0]
        # return predictions
        
        # model 是 GeneralizedRCNN
        images = self.model.preprocess_image([inputs])
        # backbone: FPN. modeling.proposal_generator.rpn.RPN.forward
        features = self.model.backbone(images.tensor)
        # RPN (StandardRPNHead, DefaultAnchorGenerator)
        proposals, _ = self.model.proposal_generator(images, features) 
        # 返回 Instances 对象, 参见
        # ps, logits = proposals[0].proposal_boxes.tensor.numpy(), proposals[0].objectness_logits.numpy()
        logits =  proposals[0].objectness_logits.numpy()
        
        # StandardROIHeads (ROIPooler, FastRCNNConvFCHead, FastRCNNOutputLayers)
        # results, _ = self.model.roi_heads(images, features, proposals, None)
        # return GeneralizedRCNN._postprocess(results, [input], images.image_sizes)
        
        ps = self.postprocess(proposals[0].proposal_boxes, image.shape[1:], height, width).tensor
        return {
            "proposals": ps,
            "logits": logits
        }

    @staticmethod
    def postprocess(ps, image_size, output_height, output_width):
        # 参见 modeling.postprocessing.detector_postprocess
        scale_x, scale_y = (
            output_width / image_size[1],
            output_height / image_size[0],
        )
        ps.scale(scale_x, scale_y)
        ps.clip(image_size)
        return ps

class CustomedVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1, instance_mode=...):
        super().__init__(img_rgb, metadata, scale, instance_mode)
    
    def draw_instance_predictions(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        colors = None
        alpha = 0.5

        self.overlay_instances(
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_proposals(self, proposals):
        # 最多展示 10 个
        boxes, logits = proposals["proposals"][:10], proposals['logits'][:10]
        labels = [str(l) for l in logits]
        self.overlay_instances(
            boxes=boxes, 
            labels=labels,
            alpha=.8)
        return self.output

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        
        self.parallel = False
        # 模型
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        运行整个模型, 绘制结果
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # vis_output = None
        
        # 运行模型
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = CustomedVisualizer(image, self.metadata, instance_mode=self.instance_mode)

        # 绘图!
        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        
            # boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            # scores = predictions.scores if predictions.has("scores") else None
            # classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
            # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        return predictions, vis_output
    
    def get_proposals(self, image):
        # 仅运行一部分的模型, 得到 proposals
        outputs = self.predictor.get_proposals(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        
        # 绘图
        visualizer = CustomedVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        vis_output = visualizer.draw_proposals(outputs)
        
        return outputs, vis_output

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    
    demo = VisualizationDemo(cfg)
    
    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        
        # 1) 绘制 outputs
        # predictions, visualized_output = demo.run_on_image(img)
        proposals, visualized_output = demo.get_proposals(img)
        
        # logger.info(
        #     "{}: {} in {:.2f}s".format(
        #         path,
        #         "detected {} instances".format(len(proposals["instances"]))
        #         if "instances" in proposals
        #         else "finished",
        #         time.time() - start_time,
        #     )
        # )
        
        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit