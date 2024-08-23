# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Dataloaders and dataset utils - 3LC integration
"""
from __future__ import annotations

import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import numpy as np
import tlc
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
from tlc.core.utils.progress import track

from utils.augmentations import Albumentations
from utils.dataloaders import InfiniteDataLoader, LoadImagesAndLabels, img2label_paths, seed_worker
from utils.general import LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, cv2
from utils.loggers.tlc.constants import TLC_COLORSTR, TLC_TRAIN_PATH, TLC_VAL_PATH
from utils.loggers.tlc.logger import TLCLogger
from utils.loggers.tlc.utils import get_names_from_yolo_table
from utils.torch_utils import torch_distributed_zero_first

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def unpack_box(bbox: dict[str, Any]) -> list[int | float]:
    return [bbox[tlc.LABEL], bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]


def tlc_table_row_to_yolo_label(row: dict[str, Any]) -> np.ndarray:
    unpacked = [unpack_box(box) for box in row[tlc.BOUNDING_BOXES][tlc.BOUNDING_BOX_LIST]]
    arr = np.array(unpacked, ndmin=2, dtype=np.float32)
    if len(unpacked) == 0:
        arr = arr.reshape(0, 5)
    return arr


def create_dataloader(
    path: tuple[str, tlc.Table, bool, bool],
    imgsz: int,
    batch_size: int,
    stride: int = 32,
    single_cls: bool = False,
    hyp: dict[str, Any] | None = None,
    augment: bool = False,
    cache: bool | str = False,
    pad: float = 0.0,
    rect: bool = False,
    rank: int = -1,
    workers: int = 8,
    image_weights: bool = False,
    quad: bool = False,
    prefix: str = "",
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[DataLoader, LoadImagesAndLabels]:
    """ Create dataloader in the 3LC integration. In addition to the standard behavior, this function also
    handles 3LC-specific arguments (zero weight exclusion and sampling weights), logging and reading of 
    other properties required by the 3LC integration logger.
    
    """
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False

    tlc_path, table, tlc_sampling_weights, tlc_exclude_zero_weight = path

    if rect and tlc_sampling_weights:
        raise ValueError("--rect is not compatible with 3LC sampling weights.")

    if tlc_path == TLC_VAL_PATH:
        TLCLogger.get_instance().register_val_args(
            path=path,
            imgsz=imgsz,
            batch_size=batch_size,
            stride=stride,
            single_cls=single_cls,
            hyp=hyp,
            augment=augment,
            cache=cache,
            pad=pad,
            rect=rect,
            rank=rank,
            workers=workers,
            image_weights=image_weights,
            quad=quad,
            prefix=prefix,
            shuffle=shuffle,
            seed=seed,
        )

    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        LOGGER.info(f"{TLC_COLORSTR}Creating dataloader for {table.dataset_name} dataset")
        dataset = TLCLoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=TLC_COLORSTR,
            table=table,
            tlc_sampling_weights=tlc_sampling_weights,
            tlc_exclude_zero_weight=tlc_exclude_zero_weight,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = (
        DataLoader if (image_weights or tlc_sampling_weights) else InfiniteDataLoader
    )  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=TLCLoadImagesAndLabels.collate_fn4 if quad else TLCLoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # Populate singleton dataloaders
    if rank in {0, -1}:
        if tlc_path == TLC_TRAIN_PATH:
            if not hasattr(TLCLogger.get_instance(), "train_loader"):
                TLCLogger.get_instance().train_loader = dataloader
        elif tlc_path == TLC_VAL_PATH:
            if not hasattr(TLCLogger.get_instance(), "val_loader"):
                TLCLogger.get_instance().val_loader = dataloader

    return dataloader, dataset


class TLCLoadImagesAndLabels(LoadImagesAndLabels):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    ]

    def __init__(
        self,
        path: Any,
        img_size: int = 640,
        batch_size: int = 16,
        augment: bool = False,
        hyp: dict[str, Any] | None = None,
        rect: bool = False,
        image_weights: bool = False,
        cache_images: bool = False,
        single_cls: bool = False,
        stride: int = 32,
        pad: float = 0.0,
        min_items: int = 0,
        prefix: str = "",
        table: tlc.Table | None = None,
        tlc_sampling_weights: bool = False,
        tlc_exclude_zero_weight: bool = False,
    ) -> None:
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.prefix = prefix
        self.albumentations = Albumentations(size=img_size) if augment else None

        if rect and tlc_sampling_weights:
            raise ValueError("Rectangular training is not compatible with 3LC sampling weights")

        self.tlc_use_sampling_weights = tlc_sampling_weights
        if tlc_sampling_weights:
            LOGGER.info(f"{prefix}Using 3LC sampling weights")

        # Get 3lc table - read the yolo image and label paths and any revisions
        self.categories = get_names_from_yolo_table(table)
        self.tlc_name = table.dataset_name
        self.tlc_table_url = table.url.to_str()

        self.sampling_weights = []
        self.im_files = []
        self.shapes = []
        self.labels = []

        num_fixed, num_corrupt = 0, 0
        msgs = []

        pbar = iter(table.table_rows)
        if RANK in {-1, 0}:
            pbar = track(pbar, description=f"Loading data from 3LC Table {table.dataset_name}/{table.url.name}", total=len(table))

        # Keep track of which example ids are in use (map from index in the yolo dataset to example id)
        self.example_ids = []
        num_ignored = 0

        for example_id, row in enumerate(pbar):
            im_file = tlc.Url(row[tlc.IMAGE]).to_absolute().to_str()

            # Fix fixable and discard corrupt images
            fixed, corrupt, msg = fix_image(im_file)
            if msg:
                msgs.append(msg)
            num_fixed += int(fixed)
            num_corrupt += int(corrupt)

            # Ignore zero weight images if tlc_exclude_zero_weight is set
            ignore = tlc_exclude_zero_weight and row[tlc.SAMPLE_WEIGHT] == 0.0
            num_ignored += int(ignore)

            discard = corrupt or ignore

            if not discard:
                self.sampling_weights.append(row[tlc.SAMPLE_WEIGHT])
                self.im_files.append(str(Path(im_file)))  # Ensure path is os.sep-delimited
                self.shapes.append((row[tlc.WIDTH], row[tlc.HEIGHT]))
                self.labels.append(tlc_table_row_to_yolo_label(row))
                self.example_ids.append(example_id)

        self.shapes = np.array(self.shapes)

        self.sampling_weights = np.array(self.sampling_weights)
        self.sampling_weights = self.sampling_weights / np.sum(self.sampling_weights)

        assert len(self.im_files) == len(self.example_ids)

        if num_fixed > 0 or num_corrupt > 0:
            LOGGER.info(f"  Fixed {num_fixed} images. Found and ignored {num_corrupt} corrupt images")

        if len(msgs) > 0:
            LOGGER.info("  " + "\n  ".join(msgs))

        if num_ignored > 0:
            LOGGER.info(f"  Excluded {num_ignored} images with zero weight")

        if len(self.im_files) == 0:
            raise ValueError(
                f"Unable to read images from {table.url}. Please check that the table is valid, and that any required aliases are set."
            )

        n = len(self.im_files)
        self.label_files = img2label_paths(
            self.im_files
        )  # .label_files is not really used in the 3LC integration, as labels are stored in the table
        self.segments = tuple([] for _ in range(n))  # TODO: Add segmentation support

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f"{prefix}{n - len(include)}/{n} images filtered from dataset")
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh
            self.sampling_weights = self.sampling_weights[include]
            self.example_ids = [self.example_ids[i] for i in include]

        # Create indices
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        self.rect_indices = None
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.rect_indices = irect.copy()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            self.sampling_weights = self.sampling_weights[irect]
            self.example_ids = [self.example_ids[i] for i in irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == "ram" and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == "disk" else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(
                enumerate(results),
                total=n,
                bar_format=TQDM_BAR_FORMAT,
                disable=LOCAL_RANK > 0,
            )
            for i, x in pbar:
                if cache_images == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    (
                        self.ims[i],
                        self.im_hw0[i],
                        self.im_hw[i],
                    ) = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{prefix}Caching images ({b / gb:.1f}GB {cache_images})"
            pbar.close()

    def resample(self, epoch: int | None = None) -> None:
        """
        Resamples the dataset using the 3LC sampling weights. Updates the .indices attribute in place.

        :param epoch: The current epoch number. If None, the seed is not set.
        """
        if self.tlc_use_sampling_weights:
            # Seed such that each process does the same sampling
            if epoch is not None:
                np.random.seed(epoch)
            LOGGER.info(f"{self.prefix}Resampling dataset for epoch {epoch}")
            # Sample indices weighted by 3LC sampling weight
            self.indices = np.random.choice(
                len(self.indices),
                size=len(self.indices),
                replace=True,
                p=self.sampling_weights,
            )


def fix_image(im_file: str) -> tuple[bool, bool, str]:
    fixed = False
    corrupt = False
    msg = ""

    # From utils/dataloaders.py
    try:
        im = Image.open(im_file)
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
                    fixed = True

    except Exception as e:
        msg = f"WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        corrupt = True

    return fixed, corrupt, msg
