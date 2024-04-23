<img src="https://3lc.ai/wp-content/uploads/2023/09/3LC-Logo_Footer.svg">

# 3LC Integration

This document outlines how to use the 3LC integration available for YOLOv5.

## About 3LC

[3LC](https://3lc.ai) is a tool which enables data scientists to improve machine learning models in a data-centric fashion. It collects per-sample predictions and metrics, allows viewing and modifying the dataset in the context of those predictions in the 3LC Dashboard, and rerunning training with the revised dataset.

3LC is free for non-commercial use.

<img src="https://docs.3lc.ai/3lc/2.2/_images/run_assign1.jpg">

## Getting Started

The integration is automatically enabled if the `3lc` package is installed in your environment. It can be installed with 
```
pip install 3lc
```

### First Time

For your first run, simply rerun your usual `train.py` command. 3LC creates `Table`s for your training and validation sets provided through `--data`, and collect metrics after the final epoch for every sample in your dataset. A new YAML file is written next to the one that was used, which can be used for later runs, more on that in [Later Runs](#later-runs).

You can then open the 3LC Dashboard to view your run!

### Later Runs

For later runs, in order to specify that you would like to continue working with the same 3LC tables, there are two ways to proceed:

#### Regular YAML file

You can keep using the same YAML file pointed to by `--data`. As long as this file does not change, the integration will resolve to the same 3LC tables and always get the latest revision for each split. The specific revisions used are logged to the console, and a line is printed stating that a 3LC YAML is printed with instructions on how to use it.

The integration uses the YAML file name to resolve to the relevant tables if they exist. Therefore, if any changes are made to the original YAML file name, this reference to the tables is lost (and new tables are instead created).

#### 3LC YAML File

For more flexibility and to explicitly select which tables to use, you can use a 3LC YAML file, like the one written during your first run. It should contain keys for the relevant splits (`train` and `val` in most cases), with values set to the 3LC Urls of the corresponding tables. Once your 3LC YAML is populated with these it will look like the following example:

```yaml
train: my_train_table
val: my_val_table
```

In order to train on different revisions, simply change the paths in the file to your desired revision.

If you would like to train on the latest revisions, you can add `:latest` to one or both of the paths, and 3LC will find the `Table`s for the latest revision in the same lineage. For the above example, with latest on both, the 3LC YAML would look like this:

```yaml
train: my_train_table:latest
val: my_val_table:latest
```

______________________________________________________________________

**NOTE**: We recommend using a 3LC YAML to specify which revisions to use, as this enables using specific revisions of the dataset, and adding `:latest` in order to use the latest table in the lineage.

______________________________________________________________________

<details>
<summary>YAML Example</summary>
<br>
As an example, let's assume that you made a new revision in the 3LC Dashboard where you edited a bounding box. You would then have the following tables:
```
my_train_table ---> Edited2BoundingBoxes (latest)
my_val_table (latest)
```

If you were to reuse the original YAML file, `Edited2BoundingBoxes` would be the latest revision of your train set and `my_val_table` the latest val set. These would be used for your run.

In order to train on a specific revision, in this case the original data, you can provide a 3LC YAML file `my_3lc_dataset.yaml` with `--data 3LC://my_3lc_dataset.yaml`, with the following content:

```yaml
train: my_train_table
val: my_val_table
```

Specifying to use the latest revisions instead can be done by adding `:latest` to one or both of these `Url`s:

```yaml
train: my_train_table:latest # resolves to the latest revision of my_train_table, which is Edited1BoundingBoxes
val: my_val_table:latest # resolves to the latest revision of my_val_table, which is my_val_table
```

</details>

### Metrics Collection Only

In some cases it is useful to collect metrics without doing any training, such as when pretrained weights already exist. This is possible by calling `val.py --task collect`.

## 3LC Settings

The integration offers a rich set of settings and features which can be set with environment variables. They allow specifying which metrics to collect, how often to collect them, and whether to use sampling weights during training. Note that some settings only apply to `train.py` or `val.py --task collect`, see the _Script_ column.

| Environment Variable                 | Default     | Comments                                                                                                                                             | Script                  |
| ------------------------------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `TLC_COLLECTION_DISABLE=true/false`  | `false`     | Whether to disable metrics collection entirely.                                                                                                      | `train.py`              |
| `TLC_COLLECTION_EPOCH_INTERVAL`      | `1`         | How frequently to collect metrics. 1 means every epoch, 2 every other and so on.                                                                     | `train.py`              |
| `TLC_COLLECTION_EPOCH_START`         | `-1`        | Which epoch to start collecting metrics (0-based). -1 means to not collect metrics, with the exception of the pass performed after training.         | `train.py`              |
| `TLC_CONF_THRES`                     | `0.1`       | Confidence threshold for predictions. Any predicted boxes with confidence less than the threshold are discarded.                                     | Both                    |
| `TLC_MAX_DET`                        | `300`       | Max number of detections per image. The lowest confidence predictions are evicted such that there are at most TLC_MAX_DET predicted boxes per image. | Both                    |
| `TLC_COLLECT_LOSS`                   | `false`     | Whether to collect loss during metrics collection. Each loss component (box, objectness and class) is recorded.                                      | `train.py`              |
| `TLC_IMAGE_EMBEDDINGS_DIM=0/2/3`     | `0`         | Dimension of reduced image embeddings. Values of 2 and 3 are supported. A value of 0 means no image embeddings are collected.                        | Both                    |
| `TLC_IMAGE_EMBEDDINGS_REDUCER`       | `umap`      | Reducer to use for embeddings collection.                                                                                                            | Both                    |
| `TLC_SAMPLING_WEIGHTS=true/false`    | `false`     | Whether to use sampling weights in training. Does not change the effective size of the dataset.                                                      | `train.py`              |
| `TLC_EXCLUDE_ZERO_WEIGHT_TRAINING`   | `false`     | Whether to exclude zero-weighted samples during training. Modifies the effective size of the dataset.                                                | `train.py`              |
| `TLC_EXCLUDE_ZERO_WEIGHT_COLLECTION` | `false`     | Whether to exclude zero-weighted samples during metrics collection. Modifies the effective size of the dataset.                                      | Both                    |
| `TLC_COLLECTION_SPLITS`              | `train,val` | List of splits to collect metrics on during metrics collection only.                                                                                 | `val.py --task=collect` |
| `TLC_COLLECTION_VAL_ONLY`            | `false`     | If set to `true`, only collect metrics on the validation set during training.                                                                        | `train.py`              |

Boolean settings can be provided in a number of ways: `true/false`, `1/0`, `y/n` and `yes/no`. Providing invalid values (or combinations of values) will either log an appropriate warning or raise an error, depending on the case.

If an environment variable starting with `TLC` is available that does not match any of the supported ones, a message is logged to the console stating that this is the case.

As an example, running the following command will train for ten epochs with sampling weights, and collect metrics (including loss and image embeddings) every other epoch:

```
TLC_IMAGE_EMBEDDINGS_DIM=2 TLC_COLLECT_LOSS=true TLC_SAMPLING_WEIGHTS=true TLC_COLLECTION_EPOCH_START 0 TLC_COLLECTION_EPOCH_INTERVAL 2 python train.py --data 3LC://my/tlc/dataset.yaml ...
```

This assumes there exists a 3LC YAML located at `my/tlc/dataset.yaml`.

### Image Embeddings

Image embeddings can be collected by setting `TLC_IMAGE_EMBEDDINGS_DIM` to 2 or 3, and are based on the output of the spatial pooling function output from the YOLOv5 architectures. Similar images, as seen by the model, tend to be close to each other in this space. In the 3LC Dashboard these embeddings can be visualized, allowing you to find similar images, find imbalances in your dataset and determine if your validation set is representative of your training data (and vice-versa).

### Loss

Loss can be collected during calls to `train.py` with `TLC_COLLECT_LOSS`. The output is the three (in the single-class case there are only two, since the classification loss is always zero) loss components computed in YOLOv5. These are useful to determine which images are challenging or easy for the model in terms of the three different loss types. The aggregate loss can be computed in the 3LC Dashboard.

## Other output

In addition to per-sample metrics, a separate metrics table is written with per-class metrics computed by YOLOv5. This is available as a second tab in the 3LC Dashboard when you open a YOLOv5 run.

When viewing all your YOLOv5 runs in the 3LC Dashboard, charts will show up with per-epoch validation metrics for each run. This allows you to follow your runs in real-time, and compare them with each other.

# Frequently Asked Questions

## What is the difference between before and after training metrics?

By default, the 3LC integration collects metrics only after training with the `best.pt` weights written by YOLOv5. These are the after training metrics.

If a starting metrics collection epoch is provided (optionally with an interval), metrics are also collected during training, this time with the exponential moving average that YOLOv5 uses for its validation passes.

## What happens if I use early stopping? Does it interfere with 3LC?

Early stopping with `--patience` can be used just like before. Unless metrics collection is disabled, final validation passes are performed over the train and validation sets after training, regardless of whether that is due to early stopping or not.
