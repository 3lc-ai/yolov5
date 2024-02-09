# YOLOv5 🚀 AGPL-3.0 license
"""3LC Logger used for training."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tlc
import torch

import val as validate
from models.experimental import attempt_load
from utils.callbacks import Callbacks
from utils.general import LOGGER
from utils.loggers.tlc.base import BaseTLCCallback
from utils.loggers.tlc.constants import TLC_COLORSTR, TLC_TRAIN_PATH
from utils.loggers.tlc.dataset import check_dataset
from utils.loggers.tlc.loss import TLCComputeLoss
from utils.loggers.tlc.settings import Settings
from utils.loggers.tlc.utils import (
    create_tlc_info_string_before_training,
    get_metrics_collection_epochs,
    get_names_from_yolo_table,
)
from utils.loss import ComputeLoss

if TYPE_CHECKING:
    from utils.loggers.tlc.model_utils import ModelEMA

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
RANK = int(os.getenv("RANK", -1))


class TLCLogger(BaseTLCCallback):
    """
    TLC Logger.

    3LC is a data management system for AI, that allows you to improve your dataset using the model predictions as a
    guide, and retrain your model with no code changes.

    The TLCLogger is a singleton class that is initialized once per call to train.py.
    """

    _instance = None

    @classmethod
    def create_instance(cls, opt=None, hyp=None) -> bool:
        """Create the singleton instance of the TLCLogger."""
        assert cls._instance is None, "TLCLogger has already been initialized."
        cls._instance = super().__new__(cls)
        cls._instance.initialize(opt=opt, hyp=hyp)
        return True

    @classmethod
    def get_instance(cls) -> TLCLogger:
        """Get the singleton instance of the TLCLogger."""
        if cls._instance is None:
            raise ValueError("TLCLogger has not been initialized yet.")
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance of the TLCLogger."""
        cls._instance = None

    def initialize(self, opt=None, hyp=None):
        self.opt = opt
        self.hyp = hyp
        self._save_dir = Path(self.opt.save_dir)  # Path to save results
        self._best_path = self._save_dir / "weights" / "best.pt"  # Path to best scoring weights

        self.current_epoch = None
        self.train_table = None
        self.val_table = None
        self.collected_for_epochs = set()  # Which epochs have we collected metrics for

        # Read 3LC specific settings from environment variables
        self._settings = self._get_settings()

        # Metrics collection epochs
        self.metrics_collection_epochs = get_metrics_collection_epochs(
            self._settings.collection_epoch_start,
            self.opt.epochs,
            self._settings.collection_epoch_interval,
            self._settings.collection_disable,
        )

        # Populate data_dict, process the provided opt.data yaml file
        self.data_dict = self.check_dataset(self.opt.data)

        self._train_validation_callbacks = Callbacks()
        self._train_validation_callbacks.register_action("on_val_batch_end", callback=self.on_val_batch_end)

        self._collecting_on = None  # Set to train or val when collecting metrics, None otherwise
        self._validation_loader_args = None  # To be populated by register_val_args when creating validation loader
        self._validation_train_loader = None  # To be created after validation loader is created
        self._model = None  # Model
        self._ema = None  # Model exponential moving average - used in validation
        self._amp = None  # Whether automatic mixed precision is enabled in training
        self.rect_indices = None
        self._reached_final_validation = False  # Whether we have reached the final validation
        self._last_validated_epoch = -1  # The last epoch we validated on

    def _get_settings(self) -> Settings:
        """Verify that 3LC settings are correct and with required dependencies."""
        self._settings = Settings.from_env()
        self._settings.verify(opt=self.opt, training=True)
        return self._settings

    def check_dataset(self, data_file: str) -> dict[str, Any]:
        """
        Check and parse the provided dataset YAML file.

        If the dataset is not registered, we register it and create the 3LC tables. A new YAML file is created, with 3LC
        URLs instead which can be used with a 3LC prefix.

        If the dataset is already registered, we use the URLs in the YAML file to get the tables.

        :param data_file: Path to the dataset YAML file.
        :returns: A data dictionary with classes, number of classes and split paths.
        """
        data_dict = check_dataset(data_file)

        self.train_table = data_dict["train"][1]
        self.val_table = data_dict["val"][1]

        names = get_names_from_yolo_table(self.train_table)
        self.label_mapping = {i: i for i in range(len(names))}

        return data_dict

    def register_amp(self, amp: bool) -> None:
        """
        Register whether amp is used.

        :param amp: Whether automatic mixed precision is used
        """
        self._amp = amp

    def register_ema(self, ema: ModelEMA) -> None:
        """Register the EMA model."""
        self._ema = ema

    def register_model(self, model: torch.nn.Module) -> None:
        """
        Register the model.

        :param model: The model to be trained.
        """
        self._model = model

    @property
    def validation_train_loader(self) -> torch.utils.data.DataLoader:
        if not self._validation_train_loader:
            self._create_validation_train_loader_from_val_loader()
        return self._validation_train_loader

    @property
    def table(self) -> tlc.Table:
        """Get the table for the current split."""
        return self.val_table if self._collecting_on == "val" else self.train_table

    @property
    def split(self) -> str:
        """Get the split for the current split."""
        return "val" if self._collecting_on == "val" else "train"

    def _create_validation_train_loader_from_val_loader(self) -> None:
        """Create a dataloader to use for validation on the train split, copying settings from the validation loader."""
        # Do not use sampling weights when creating the validation train loader
        from utils.loggers.tlc.dataloaders import create_dataloader

        # Create a dataloader for the train set, using the same settings as the validation loader
        validation_train_loader_args = self._validation_loader_args
        validation_train_loader_args["path"] = (TLC_TRAIN_PATH + "_validate", self.train_table, False)
        self._validation_train_loader = create_dataloader(**validation_train_loader_args)[0]

    def register_val_args(self, **kwargs: Any) -> None:
        """Register arguments for the validation dataloader, to be used when creating the validation train loader."""
        self._validation_loader_args = kwargs

    def on_train_start(self) -> None:
        if not self._settings.collection_disable:
            # Create a 3LC run and log run parameters
            self.run = tlc.init(project_name=self.train_table.project_name)

            parameters = {k: v for k, v in vars(self.opt).items() if k != "hyp"}
            parameters["evolve_population"] = str(parameters["evolve_population"])
            parameters.update(self.hyp)
            self.run.set_parameters(parameters=parameters)

            self._model.hyp = self.hyp  # Required for losses
            self._unreduced_loss_fn = TLCComputeLoss(self._model)
            self._loss_fn = ComputeLoss(self._model)

        # Print 3LC information
        tlc_mc_string = create_tlc_info_string_before_training(
            self.metrics_collection_epochs,
            disable=self._settings.collection_disable,
        )
        LOGGER.info(TLC_COLORSTR + tlc_mc_string)

    def on_train_epoch_start(self, epoch: int) -> None:
        self.current_epoch = epoch

    def on_train_epoch_end(self) -> None:
        self._ema.ema.collect_embeddings = self._settings.image_embeddings_dim > 0 and self.should_collect_metrics()

    def on_fit_epoch_end(self, vals: list[torch.Tensor | float], epoch: int) -> None:
        """
        Called after each epoch in fit, to store the validation outputs for the epoch.

        :param vals: The validation outputs for the epoch mloss + val metrics + lr [loss, loss, loss, mp, mr, map50,
            map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, lr]
        :param epoch: The current epoch
        """
        if self.should_collect_metrics():
            self.collected_for_epochs.add(self.current_epoch)

        # Store the validation outputs for the epoch
        val_metrics = {
            "val precision": vals[3],  # mp
            "val recall": vals[4],  # mr
            "val mAP50": vals[5],  # map50
            "val mAP50-95": vals[6],  # map
            "val loss": vals[7],  # *(loss.cpu() / len(dataloader)
            "learning rate": vals[-1],
            "epoch": epoch,
        }
        self.run.add_output_value(val_metrics)

    def on_train_end(self, results: list[int | float]) -> None:
        """
        Reduce any embeddings and write final per-class metrics to the run before closing.

        :param results: The final aggregate metrics provided by YOLOv5.
        """
        if self._settings.image_embeddings_dim != 0:
            self.run.reduce_embeddings_by_example_table_url(
                self.val_table.url,
                method=self._settings.image_embeddings_reducer,
                n_components=self._settings.image_embeddings_dim,
            )

        self._log_final_metrics(results)

        tlc.close()

    def _log_final_metrics(self, results: list[int | float]) -> None:
        """Write the final aggregate metrics to the run."""
        pass
        # final_metrics = {
        #     'val_precision': results[0],  # mp
        #     'val_recall': results[1],  # mr
        #     'val_mAP50': results[2],  # map50
        #     'val_mAP': results[3],  # map
        #     'val_loss': results[4], }  # *(loss.cpu() / len(dataloader)

        # self.run.add_output_value(final_metrics)

    def should_collect_metrics(self) -> bool:
        """
        Whether we should collect metrics for the current epoch.

        :returns: Whether we should collect metrics for the current epoch
        """

        # If collection is disabled, we should not collect
        if self._settings.collection_disable:
            return False

        # We should collect if the current epoch is in the list of epochs to collect for
        return (self.current_epoch in self.metrics_collection_epochs) or self._reached_final_validation

    def _collect_on_train(self) -> None:
        if not self._settings.collection_val_only:
            self._collecting_on = "train"

            # Prepare logger for metrics collection
            # Then call validate.run with this logger as a callback (this will clean up the metrics etc.)
            self.metrics_writer = tlc.MetricsWriter(
                run_url=self.run.url,
                dataset_url=self.train_table.url,
                dataset_name=self.train_table.dataset_name,
                override_column_schemas=self.metrics_schema,
            )
            self.example_ids_for_batch = list(
                tlc.batched_iterator(
                    range(len(self.train_table)), batch_size=self._validation_loader_args["batch_size"]
                )
            )

            self.rect_indices = self.validation_train_loader.dataset.rect_indices

            model = (
                self._ema.ema
                if not self._reached_final_validation
                else attempt_load(self._best_path, next(self._ema.ema.parameters()).device).half()
            )
            model.collect_embeddings = self._settings.image_embeddings_dim > 0 and self.should_collect_metrics()

            LOGGER.info(TLC_COLORSTR + "Collecting metrics on train and val sets:")
            validate.run(
                self.data_dict,
                batch_size=self._validation_loader_args["batch_size"],
                imgsz=self.opt.imgsz,
                half=self._amp if not self._reached_final_validation else True,
                model=model,
                single_cls=self.opt.single_cls,
                dataloader=self.validation_train_loader,
                plots=self._reached_final_validation,
                save_dir=self._save_dir if self._reached_final_validation else Path(""),
                callbacks=self._train_validation_callbacks,
                compute_loss=self._loss_fn,
            )

    def on_val_start(self) -> None:
        # Check if we have reached final validation (either early stopping or last epoch)
        self._reached_final_validation = self.current_epoch == self._last_validated_epoch
        if not self._reached_final_validation:
            self._last_validated_epoch = self.current_epoch

        if not self.should_collect_metrics():
            return

        # When we get here with _collecting_on set to train, we are in the call to validate.run from
        # TLCLogger.on_val_start, and we don't want to do anything here.
        if self._collecting_on == "train":
            return

        # Collect metrics on train set if enabled
        self._collect_on_train()

        # Then we go ahead with the already started validation (metrics collection) on the val set
        # We then let things run its course.
        self._collecting_on = "val"
        self.rect_indices = self.val_loader.dataset.rect_indices

        if self._amp:
            self._ema.ema.half()

        # Val set validation is called from train.py already, so we let that happen by itself.
        self.metrics_writer = tlc.MetricsWriter(
            run_url=self.run.url,
            dataset_url=self.val_table.url,
            dataset_name=self.val_table.dataset_name,
            override_column_schemas=self.metrics_schema,
        )
        self.example_ids_for_batch = list(
            tlc.batched_iterator(range(len(self.val_table)), batch_size=self._validation_loader_args["batch_size"])
        )

    def on_val_batch_end(self, batch_i, images, targets, paths, shapes, outputs, train_out) -> None:
        """
        Called after each batch in validation.

        :param batch_i: The batch index
        :param images: The images in the batch
        :param targets: The targets in the batch
        :param paths: The image paths in the batch
        :param shapes: The image shape data for the batch, including padding/resize data
        :param outputs: The model outputs for the batch
        :param train_out: The model training outputs for the batch (data to compute loss)
        """
        if not self.should_collect_metrics():
            return

        self._save_batch_metrics(batch_i, images, targets, paths, shapes, outputs, train_out, epoch=self.current_epoch)

        # Find out if last batch, and if so, flush the metrics writer
        if batch_i == len(self.example_ids_for_batch) - 1:
            input_table = self.train_table if self._collecting_on == "train" else self.val_table
            self.flush_metrics_writer(input_table=input_table)
