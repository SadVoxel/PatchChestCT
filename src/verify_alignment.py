"""
verify_alignment.py  —  PatchChestCT alignment checker
=======================================================
Overlays patch-level annotations on the preprocessed CT volume and saves a
PNG montage.  Run this after setting up your data to confirm that annotations
align correctly with the anatomy.

Usage
-----
  python verify_alignment.py \\
      --ct  /path/to/train_preprocessed/train_1003/train_1003a/train_1003_a_1.npz \\
      --ann annotations-train/train_1003_a_1/ \\
      --out alignment_check.png

The output image shows 24 reference axial slices (the same view the annotators
used) with coloured patch outlines for each annotated abnormality.

Coordinate convention
---------------------
PatchChestCT annotations are stored in a **canonical chest-CT display frame**
that matches standard radiological viewing conventions:

  * Axial view: patient's left is on the right side of the image
    (i.e. the image is as seen from below, the standard radiology convention).
  * The 24 slices run from apex (slice 0) to base (slice 23).
  * Row 0 is at the top (anterior), row 11 at the bottom (posterior).

This display frame is derived from the raw CT-RATE volumes via a deterministic
preprocessing pipeline (see `_preprocess_to_display_space()` below).  The
annotation grid maps directly onto this display frame:

    annotation ann[z, y, x] = 1
    ↕
    display_volume[4*z : 4*(z+1), 16*y : 16*(y+1), 16*x : 16*(x+1)]

where display_volume has shape (96, 192, 192) — (depth, height, width).

If the coloured boxes in the output PNG land on the correct anatomy (e.g.
arterial calcification boxes appear over the aorta, consolidation boxes appear
in the lung parenchyma), your pipeline is correctly aligned.
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom


# ── colour palette (one per disease) ─────────────────────────────────────────
PALETTE = {
    "arterial_wall_calcification":  "red",
    "coronary_wall_calcification":  "orange",
    "pericardial_effusion":         "yellow",
    "hiatal_hernia":                "lime",
    "lymphadenopathy":              "cyan",
    "atelectasis":                  "blue",
    "lung_opacity":                 "white",
    "consolidation":                "magenta",
    "bronchiectasis":               "pink",
}


def _preprocess_to_display_space(ct_path: str) -> np.ndarray:
    """
    Load a raw CT-RATE volume (.npz, arr_0) and transform it to the canonical
    display space used during annotation.

    Pipeline (applied to the raw (D, H, W) array):
      1. Multiply by 1000 to restore Hounsfield units.
      2. transpose(1, 2, 0)  →  (H, W, D)
      3. rot90(k=-1, axes=(0,1))  — 90° clockwise in the axial plane
         This step aligns the image with standard radiological orientation
         (patient's left on the right, anterior at top).
      4. Center-crop / zero-pad to (192, 192, 96).
      5. transpose(2, 0, 1)  →  (96, 192, 192)  i.e. (depth, H, W)
      6. flip along W axis (axis=2) — left-right flip to match display space.
      7. Subsample depth: take every 4th slice starting at index 2
         → 24 reference slices.

    Returns
    -------
    display : np.ndarray, shape (96, 192, 192), float32
        Full-resolution display-space volume.
    """
    arr = np.load(ct_path)["arr_0"].astype(np.float32) * 1000.0

    # Step 1-2: (D,H,W) → (H,W,D) → rotate 90° CW in axial plane
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.rot90(arr, k=-1, axes=(0, 1))

    # Step 3: center crop/pad to (192, 192, 96)
    target = (192, 192, 96)
    for axis, tgt in enumerate(target):
        cur = arr.shape[axis]
        if cur >= tgt:
            start = (cur - tgt) // 2
            slices = [slice(None)] * 3
            slices[axis] = slice(start, start + tgt)
            arr = arr[tuple(slices)]
        else:
            pad = [(0, 0)] * 3
            before = (tgt - cur) // 2
            after  = tgt - cur - before
            pad[axis] = (before, after)
            arr = np.pad(arr, pad, mode="constant", constant_values=0)

    # Step 4-5: (H,W,D) → (D,H,W), then flip W
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.flip(arr, axis=2).copy()

    return arr   # (96, 192, 192)


def _load_annotations(ann_dir: str) -> dict:
    """Return {disease_name: (24,12,12) bool array} for files found in ann_dir."""
    anns = {}
    for npz_path in sorted(glob.glob(os.path.join(ann_dir, "*.npz"))):
        disease = os.path.splitext(os.path.basename(npz_path))[0]
        anns[disease] = np.load(npz_path)["arr_0"].astype(bool)
    return anns


def _apply_window(arr: np.ndarray, wl: float = -400, ww: float = 1200) -> np.ndarray:
    """Apply lung-window and normalise to [0, 1]."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


def visualise(ct_path: str, ann_dir: str, out_path: str) -> None:
    print(f"Loading CT: {ct_path}")
    vol = _preprocess_to_display_space(ct_path)   # (96, 192, 192)
    ref_slices = vol[2::4]                         # (24, 192, 192)  — same 24 used in annotation

    print(f"Loading annotations from: {ann_dir}")
    anns = _load_annotations(ann_dir)
    if not anns:
        print("  (no annotation files found — drawing CT only)")

    # Each patch covers 16×16 pixels in the 192×192 frame
    PATCH_H = 16
    PATCH_W = 16

    nrows, ncols = 4, 6   # 24 slices in a 4×6 grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.patch.set_facecolor("black")

    for idx in range(24):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor("black")
        ax.imshow(_apply_window(ref_slices[idx]), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"slice {idx}", color="white", fontsize=7)
        ax.axis("off")

        for disease, ann in anns.items():
            colour = PALETTE.get(disease, "white")
            for r in range(12):
                for c in range(12):
                    if ann[idx, r, c]:
                        rect = mpatches.Rectangle(
                            (c * PATCH_W, r * PATCH_H),
                            PATCH_W, PATCH_H,
                            linewidth=1.2, edgecolor=colour,
                            facecolor="none"
                        )
                        ax.add_patch(rect)

    # Legend
    handles = [
        mpatches.Patch(edgecolor=PALETTE.get(d, "white"), facecolor="none",
                       label=d.replace("_", " "), linewidth=1.5)
        for d in anns
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   fontsize=8, framealpha=0.7,
                   bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        f"PatchChestCT alignment check\n{os.path.basename(ct_path)}",
        color="white", fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="black", pad_inches=0.1)
    plt.close()
    print(f"Saved → {out_path}")
    print("\nHow to interpret:")
    print("  - Each coloured box should overlap the correct anatomy.")
    print("  - Red   = arterial wall calcification  (expect: aorta / major vessels)")
    print("  - Orange= coronary wall calcification  (expect: coronary arteries)")
    print("  - White = lung opacity                 (expect: lung parenchyma)")
    print("  - Magenta= consolidation               (expect: lung consolidation)")
    print("  - Blue  = atelectasis                  (expect: collapsed lung)")
    print("  If boxes are in clearly wrong regions, check your CT preprocessing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify PatchChestCT annotation alignment")
    parser.add_argument("--ct",  required=True,
                        help="Path to preprocessed CT .npz file (CT-RATE format)")
    parser.add_argument("--ann", required=True,
                        help="Path to annotation directory (e.g. annotations-train/train_1003_a_1/)")
    parser.add_argument("--out", default="alignment_check.png",
                        help="Output PNG path (default: alignment_check.png)")
    args = parser.parse_args()

    if not os.path.isfile(args.ct):
        raise FileNotFoundError(f"CT file not found: {args.ct}")
    if not os.path.isdir(args.ann):
        raise FileNotFoundError(f"Annotation directory not found: {args.ann}")

    visualise(args.ct, args.ann, args.out)
