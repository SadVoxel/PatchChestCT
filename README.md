# CT-RATE Patch

Welcome to the reference implementation for CT-RATE-patch. The project ships two training pipelines built on 3D ResNet-18 backbones and Monai-based data loaders.

You can obtain the CT-RATE dataset and annotations from the original CT-RATE release (follow the authors’ licensing terms) and place the processed volumes under the paths expected by `dataset_train.py` and `dataset_valid.py`.

## Baselines

Prepare the data folders:

- Put training volumes in `./train_preprocessed`
- Put validation volumes in `./valid_preprocessed`
- Store train annotations in `./annotations_train`
- Store validation annotations in `./annotations_valid`
- Drop the CSV report metadata and multi-label CSVs under `./`

Update the paths in `train_grounding.py`, `train_grounding_weak.py`, `dataset_train.py`, and `dataset_valid.py` to match the locations above.

### Training

Fully-supervised (patch annotations):

```bash
python train_grounding.py
```

Weak supervision (study-level labels):

```bash
python train_grounding_weak.py
```

Both scripts log to Weights & Biases by default

