# PatchChestCT

Reference implementation for the **PatchChestCT** dataset — a large-scale 3D patch-level annotation dataset for localizing multiple abnormalities in chest CT.

> **Dataset:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16812648.svg)](https://doi.org/10.5281/zenodo.16812648)  


---

## Dataset Overview

PatchChestCT provides patch-level binary labels for **9 clinically significant abnormalities** across **2,201 chest CT volumes** derived from the [CT-RATE dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

Each annotation is stored as a binary NumPy array (`.npz`) with shape **(24, 12, 12)**, corresponding to the 3D patch grid of the CT volume after standard preprocessing.

**Annotated abnormalities:**

| Label file | Abnormality |
|---|---|
| `arterial_wall_calcification.npz` | Arterial wall calcification |
| `coronary_wall_calcification.npz` | Coronary artery calcification |
| `pericardial_effusion.npz` | Pericardial effusion |
| `hiatal_hernia.npz` | Hiatal hernia |
| `lymphadenopathy.npz` | Lymphadenopathy |
| `atelectasis.npz` | Atelectasis |
| `lung_opacity.npz` | Lung opacity |
| `consolidation.npz` | Consolidation |
| `bronchiectasis.npz` | Bronchiectasis |

---

## Obtaining the Data

### Step 1 — Get CT-RATE source volumes

The source CT volumes are **not included** in the PatchChestCT Zenodo deposit. You must obtain them independently from the official CT-RATE repository:

1. Go to: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
2. Request gated access and agree to the Data Usage Agreement (DUA).
3. Download and preprocess the volumes following the CT-RATE documentation.

### Step 2 — Download PatchChestCT annotations

Download `annotations-train.zip` and `annotations-valid.zip` from Zenodo:

```
https://doi.org/10.5281/zenodo.16812648
```

### Step 3 — Align annotations with source volumes

Each subdirectory in the annotation zips is named using the **original CT-RATE volume identifier** (e.g., `train_1003_a_1/`), which directly corresponds to the CT-RATE naming convention.

```python
import numpy as np
import os

def load_annotation(annotation_dir, volume_id, abnormality):
    """Load a patch-level annotation for a given volume and abnormality."""
    path = os.path.join(annotation_dir, volume_id, f"{abnormality}.npz")
    if not os.path.exists(path):
        return None  # abnormality not present in this volume
    return np.load(path)["arr_0"]  # shape: (24, 12, 12), dtype: bool

# Example
ann = load_annotation("./annotations-train", "train_1003_a_1", "arterial_wall_calcification")
print(ann.shape)   # (24, 12, 12)
print(ann.sum())   # number of positive patches
```

---

## Baselines

### Data preparation

```
project_root/
├── train_preprocessed/     # CT-RATE training volumes (obtain from CT-RATE)
├── valid_preprocessed/     # CT-RATE validation volumes
├── annotations-train/      # PatchChestCT train annotations (from Zenodo)
├── annotations-valid/      # PatchChestCT valid annotations (from Zenodo)
├── train_reports.csv
├── dataset_radiology_text_reports_validation_reports.csv
├── dataset_multi_abnormality_labels_train_predicted_labels.csv
└── dataset_multi_abnormality_labels_valid_predicted_labels.csv
```

Update the path variables at the top of `src/train_grounding.py`, `src/train_grounding_weak.py`, `src/dataset_train.py`, and `src/dataset_valid.py` if your paths differ.

### Training

Fully-supervised (patch-level annotations), with selectable backbone:

```bash
python src/train_grounding.py --backbone r3d18     # R3D-18, default
python src/train_grounding.py --backbone swin3d_t  # Swin3D-T
python src/train_grounding.py --backbone mvit       # MViT-v2-S
```

Weakly supervised (study-level image labels only), with selectable backbone and method:

```bash
python src/train_grounding_weak.py --backbone r3d18 --method noisyor     # NoisyOR + R3D-18, default
python src/train_grounding_weak.py --backbone swin3d_t --method noisyor  # NoisyOR + Swin3D-T
python src/train_grounding_weak.py --backbone mvit --method noisyor      # NoisyOR + MViT-v2-S
python src/train_grounding_weak.py --backbone r3d18 --method gradcam     # Grad-CAM (r3d18 only)
```

Both scripts log to [Weights & Biases](https://wandb.ai) by default.

---

## License

The PatchChestCT annotations are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), consistent with the CT-RATE Data Usage Agreement. The source CT volumes remain subject to the CT-RATE DUA and must be obtained via the official CT-RATE gated access procedure.

---


