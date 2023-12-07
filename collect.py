# YOLOv5 🚀 AGPL-3.0 license
"""
Collect 3LC metrics for a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python collect.py --weights yolov5s.pt --data coco128.yaml --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import tlc

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from tlc.client.torch.metrics.collect import collect_metrics
from tlc.core.builtins.constants.column_names import BOUNDING_BOXES

from models.common import DetectMultiBackend
from utils.general import LOGGER, check_dataset, check_img_size, check_requirements, check_yaml, colorstr, print_args
from utils.tlc_integration import (TLCComputeLoss, create_dataloader, get_or_create_tlc_table,
                                   tlc_create_metrics_collectors)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        batch_size=1,  # batch size TODO: Support batch size > 1
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        tlc_iou_thres=0.3,  # 3LC Metrics collection IoU threshold
        max_det=300,  # maximum detections per image
        split='val',  # Split to collect metrics for
        device='',  # cuda device, i.e. 0 or cpu (only single device supported)
        workers=8,  # max dataloader workers
        single_cls=False,  # treat as single-class dataset
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        stride=None,  # stride (from training)
        epoch=None,  # epoch (use when training)
        model=None,  # model instance
        table=None,  # table (from training)
        tlc_revision_url='',  # 3LC revision URL to use for metrics collection
        tlc_image_embeddings_dim=0,  # Dimension of image embeddings (2 or 3). Default is 0, which means no image embeddings are used.
        compute_loss=None,  # ComputeLoss instance (from training)
        collect_loss=False,  # Compute and collect loss for each image during metrics collection
):

    # Initialize/load model and set device
    training = model is not None

    if tlc_image_embeddings_dim not in (0, 2, 3):
        raise ValueError(f'Invalid value for tlc_image_embeddings_dim: {tlc_image_embeddings_dim}')
    if tlc_image_embeddings_dim in (2, 3):
        # We need to ensure we have UMAP installed
        try:
            import umap  # noqa: F401
        except ImportError:
            raise ValueError('Missing UMAP dependency, run `pip install umap-learn` to enable embeddings collection.')

    if training:  # called by train.py
        # Check for required args
        if any(v is None for v in (epoch, table)):
            raise ValueError('When training, epoch and table must be passed')

        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
        model.collecting = tlc_image_embeddings_dim > 0

    else:  # called directly
        # Check for required args
        if any(v is None for v in (weights, data, split)):
            raise ValueError('When not training, model weights, data and split must be passed')

        device = select_device(device, batch_size=batch_size)

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half, fuse=False)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        if data:
            check_dataset(data)  # check
        table = get_or_create_tlc_table(
            yolo_yaml_file=data,
            split=split,
            revision_url=tlc_revision_url,
        )
        if collect_loss:
            m = model.model.model[-1]
            compute_loss = TLCComputeLoss('cpu', model.model.hyp, m.stride, m.na, m.nc, m.nl,
                                          m.anchors)  # DetectMultiBackend holds a DetectionModel, which has hyp
        else:
            compute_loss = None
        run = tlc.init(project_name=table.project_name)  # Only create a run when called directly

    # Ensure table is in collecting metrics mode
    table.collecting_metrics = True

    # Setup dataloader
    dataloader = create_dataloader(
        data,  # Not really used
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=0.5,
        rect=False,
        workers=workers,
        prefix=colorstr(f'collect-{split}: '),
        table=table,
    )[0]

    # Verify dataset classes
    categories = table.get_value_map_for_column(BOUNDING_BOXES) if not training else dataloader.dataset.categories
    nc = 1 if single_cls else len(categories)  # number of classes

    if not training and pt and not single_cls:  # check --weights are trained on --data
        ncm = model.model.nc
        assert ncm == nc, (f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} '
                           f'classes). Pass correct combination of --weights and --data that are trained together.')
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        model.model.collecting = tlc_image_embeddings_dim > 0

    # Configure
    model.eval()

    # Set up metrics collectors
    metrics_collectors = tlc_create_metrics_collectors(model=model,
                                                       names=categories,
                                                       conf_thres=conf_thres,
                                                       nms_iou_thres=iou_thres,
                                                       max_det=max_det,
                                                       iou_thres=tlc_iou_thres,
                                                       compute_embeddings=tlc_image_embeddings_dim > 0,
                                                       compute_loss=compute_loss)

    # If half precision, update metrics collector models to this
    # if half:
    #     for metrics_collector in metrics_collectors:
    #         metrics_collector.model.half()

    # Collect metrics
    collect_metrics(
        table=dataloader.dataset,
        metrics_collectors=metrics_collectors,
        constants={'epoch': epoch} if epoch is not None else {},
        dataset_name=dataloader.dataset.tlc_name,
        dataset_url=dataloader.dataset.tlc_table_url,
        dataloader_args={
            'batch_size': batch_size,
            'collate_fn': dataloader.collate_fn,
            'num_workers': workers, },
    )

    # Finish up
    if training:
        model.float()
        model.train()
        model.collecting = False
        table.collecting_metrics = False
        return None, dataloader

    else:
        if tlc_image_embeddings_dim in (2, 3):
            run.reduce_embeddings_per_dataset(n_components=opt.tlc_image_embeddings_dim)
        tlc.close()
        return run, dataloader


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--split', type=str, default='val', help='Split to collect metrics for')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=1, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for metrics collection. Defaults to 4.')
    # 3LC args
    parser.add_argument('--tlc-iou-thres',
                        type=float,
                        default=0.3,
                        help='IoU threshold for 3LC to consider a prediction a match')
    parser.add_argument('--tlc-revision-url',
                        type=str,
                        default='',
                        help='URL to the revision of the 3LC dataset to collect metrics for')
    parser.add_argument('--tlc-image-embeddings-dim',
                        type=int,
                        default=0,
                        help='Dimension of image embeddings (2 or 3). Defaults to 0, corresponding to no embeddings.')
    parser.add_argument('--tlc-collect-loss',
                        dest='collect_loss',
                        action='store_true',
                        help='Collect loss for each image during metrics collection.')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
