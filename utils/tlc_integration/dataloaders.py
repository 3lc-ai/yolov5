# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Dataloaders and dataset utils - 3LC integration
"""
import os
from collections import Counter
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import tlc
import torch
from PIL import Image, ImageOps
from tlc.core.builtins.constants.column_names import BOUNDING_BOXES, HEIGHT, IMAGE, SAMPLE_WEIGHT, WIDTH
from tlc.core.url import Url
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations
from utils.general import LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, colorstr, cv2
from utils.torch_utils import torch_distributed_zero_first

from ..dataloaders import InfiniteDataLoader, LoadImagesAndLabels, img2label_paths, seed_worker
from .utils import tlc_table_row_to_yolo_label

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = (str(os.getenv('PIN_MEMORY', True)).lower() == 'true')  # global pin_memory for dataloaders


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix='',
    shuffle=False,
    seed=0,
    table=None,
    tlc_sampling_weights=False,
):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False

    assert table is not None, 'table must be provided'
    tlc_prefix = colorstr('3LC: ')

    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        LOGGER.info(f'{tlc_prefix}Creating dataloader for {table.dataset_name} dataset')
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
            prefix=tlc_prefix,
            table=table,
            tlc_sampling_weights=tlc_sampling_weights,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = (None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle))
    loader = (DataLoader if (image_weights or tlc_sampling_weights) else InfiniteDataLoader
              )  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return (
        loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            pin_memory=PIN_MEMORY,
            collate_fn=TLCLoadImagesAndLabels.collate_fn4 if quad else TLCLoadImagesAndLabels.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        dataset,
    )


class TLCLoadImagesAndLabels(LoadImagesAndLabels):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4, ]

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        min_items=0,
        prefix='',
        table=None,
        tlc_sampling_weights=False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = (self.augment and not self.rect)  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.prefix = prefix
        self.albumentations = Albumentations(size=img_size) if augment else None

        if rect and tlc_sampling_weights:
            raise ValueError('Rectangular training is not compatible with 3LC sampling weights')

        self.tlc_use_sampling_weights = tlc_sampling_weights
        if tlc_sampling_weights:
            LOGGER.info(f'{prefix}Using 3LC sampling weights')

        # Get 3lc table - read the yolo image and label paths and any revisions
        self.categories = table.get_value_map_for_column(BOUNDING_BOXES)
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
            pbar = tlc.track(pbar, description=f'Loading data from 3LC Table {table.url.name}', total=len(table))

        for row in pbar:
            im_file = Url(row[IMAGE]).to_absolute().to_str()
            fixed, corrupt, msg = fix_image(im_file)
            if msg:
                msgs.append(msg)
            num_fixed += int(fixed)
            num_corrupt += int(corrupt)

            # Ignore corrupt images when training or validating
            # Don't ignore when collecting metrics since the dataset length will change
            if not corrupt or table.collecting_metrics:
                self.sampling_weights.append(row[SAMPLE_WEIGHT])
                self.im_files.append(str(Path(im_file)))  # Ensure path is os.sep-delimited
                self.shapes.append((row[WIDTH], row[HEIGHT]))
                self.labels.append(tlc_table_row_to_yolo_label(row))

        self.shapes = np.array(self.shapes)
        self.sampling_weights = np.array(self.sampling_weights)
        self.sampling_weights = self.sampling_weights / np.sum(self.sampling_weights)

        if num_fixed > 0 or num_corrupt > 0:
            LOGGER.info(f'Fixed {num_fixed} images. Found and ignored {num_corrupt} corrupt images')

        if len(msgs) > 0:
            LOGGER.info('\n'.join(msgs))

        n = len(self.im_files)
        self.label_files = img2label_paths(
            self.im_files)  # .label_files is not really used in the 3LC integration, as labels are stored in the table
        self.segments = tuple([] for _ in range(n))  # TODO: Add segmentation support

        # Filter images
        if min_items:
            include = (np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int))
            LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

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
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            self.sampling_weights = self.sampling_weights[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = (np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride)

        # Cache images into RAM/disk for faster training
        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = (self.cache_images_to_disk if cache_images == 'disk' else self.load_image)
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(
                enumerate(results),
                total=n,
                bar_format=TQDM_BAR_FORMAT,
                disable=LOCAL_RANK > 0,
            )
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    (
                        self.ims[i],
                        self.im_hw0[i],
                        self.im_hw[i],
                    ) = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()

    @staticmethod
    def _print_histogram_with_buckets(data, num_buckets=10) -> None:
        # Bucketing data
        bucket_size = (max(data) - min(data)) // num_buckets
        bucketed_data = [x // bucket_size * bucket_size for x in data]
        counter = Counter(bucketed_data)
        max_count = max(counter.values())

        for value in range(min(bucketed_data), max(bucketed_data) + bucket_size, bucket_size):
            count = counter.get(value, 0)
            bar = '*' * int(count / max_count * 50)  # Scale the bar length
            LOGGER.info(f'{value:5} - {value + bucket_size - 1:5} | {bar}')

    def resample(self, epoch=None):
        if self.tlc_use_sampling_weights:
            # Seed such that each process does the same sampling
            if epoch is not None:
                np.random.seed(epoch)
            LOGGER.info(f'{self.prefix}Resampling dataset for epoch {epoch}')
            # Sample indices weighted by 3LC sampling weight
            self.indices = np.random.choice(
                len(self.indices),
                size=len(self.indices),
                replace=True,
                p=self.sampling_weights,
            )


def fix_image(im_file):
    fixed = False
    corrupt = False
    msg = ''

    # From utils/dataloaders.py
    try:
        im = Image.open(im_file)
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'
                    fixed = True

    except Exception as e:
        msg = f'WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        corrupt = True

    return fixed, corrupt, msg
