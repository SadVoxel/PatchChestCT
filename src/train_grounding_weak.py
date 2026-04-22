"""
Weakly supervised grounding for PatchChestCT.

Usage:
  python train_grounding_weak.py [--backbone {r3d18,swin3d_t,mvit}] [--method {noisyor,gradcam}]

Backbones:
  r3d18    : R3D-18,    latent 512  (default)
  swin3d_t : Swin3D-T,  latent 768
  mvit     : MViT-v2-S, latent 512

Methods:
  noisyor : NoisyOR aggregation with study-level BCE  (default)
  gradcam : image-level classifier + Grad-CAM heatmaps
"""
import argparse
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import wandb
import torchvision.models.video as models
from dataset_train import CTReportDatasetTrain
from dataset_valid import CTReportDatasetInfer

# ── data paths ───────────────────────────────────────────────────────────────
data_train         = "train_preprocessed"
data_valid         = "valid_preprocessed"
reports_file_train = "train_reports.csv"
reports_file_valid = "dataset_radiology_text_reports_validation_reports.csv"
labels_train       = "dataset_multi_abnormality_labels_train_predicted_labels.csv"
labels_valid       = "dataset_multi_abnormality_labels_valid_predicted_labels.csv"

SELECTED_IDX = [1, 3, 4, 5, 6, 8, 10, 15, 16]

ALL_PATHOLOGIES = [
    'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
    'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
    'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity',
    'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
    'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
    'Interlobular septal thickening',
]

# ── arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', default='r3d18',
                    choices=['r3d18', 'swin3d_t', 'mvit'],
                    help='Backbone architecture (default: r3d18)')
parser.add_argument('--method', default='noisyor',
                    choices=['noisyor', 'gradcam'],
                    help='Weak supervision method (default: noisyor)')
args = parser.parse_args()

_BACKBONE_DEFAULTS = {
    'r3d18':    {'latent_dim': 512, 'batchsize': 10},
    'swin3d_t': {'latent_dim': 768, 'batchsize': 6},
    'mvit':     {'latent_dim': 512, 'batchsize': 6},
}

config = {
    'backbone':    args.backbone,
    'method':      args.method,
    'lr':          1e-4,
    'wd':          1e-4,
    'epochs':      20,
    'print_every': 500,
    'save_every':  500,
    'smoothing':   0.0,
    'pretrained':  True,
    'save_dir':    f'checkpoints/{args.backbone}-{args.method}/',
    **_BACKBONE_DEFAULTS[args.backbone],
}

wandb.init(project='PatchChestCT',
           name=f'{args.backbone}-{args.method}',
           config=config)

# ── encoder construction ─────────────────────────────────────────────────────
_hook_storage = {}

def _build_encoder():
    if args.backbone == 'r3d18':
        enc = models.r3d_18(pretrained=config['pretrained'])
        enc.stem[0] = nn.Conv3d(1, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2),
                                padding=(0, 0, 0), bias=False)
        enc.avgpool = nn.Identity()
        enc.fc = nn.Identity()
        return enc

    if args.backbone == 'swin3d_t':
        enc = models.swin3d_t(
            weights=models.Swin3D_T_Weights.DEFAULT if config['pretrained'] else None)
        enc.patch_embed.proj = nn.Conv3d(1, 96, kernel_size=(16, 2, 2), stride=(16, 2, 2))
        enc.norm = nn.Identity()
        enc.head = nn.Identity()
        def hook_fn(module, input, output):
            _hook_storage['swin_feat'] = output
        enc.features[-1].register_forward_hook(hook_fn)
        return enc

    # mvit
    from torchvision.models.video.mvit import MViT, MSBlockConfig
    num_heads = [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8]
    input_ch  = [64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512]
    output_ch = [64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512]
    kernel_q  = [[3, 3, 3]] * 16
    kernel_kv = [[3, 3, 3]] * 16
    stride_q  = [[1,1,1],[1,2,2],[1,1,1],[1,2,2],[1,1,1],[1,1,1],[1,1,1],[1,1,1],
                 [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,2,2],[1,1,1]]
    stride_kv = [[1,8,8],[1,4,4],[1,4,4],[1,2,2],[1,2,2],[1,2,2],[1,2,2],[1,2,2],
                 [1,2,2],[1,2,2],[1,2,2],[1,2,2],[1,2,2],[1,2,2],[1,1,1],[1,1,1]]
    block_setting = [
        MSBlockConfig(
            num_heads=num_heads[i], input_channels=input_ch[i],
            output_channels=output_ch[i], kernel_q=kernel_q[i],
            kernel_kv=kernel_kv[i],
            stride_q=stride_q[i] if i < len(stride_q) else [1, 1, 1],
            stride_kv=stride_kv[i],
        )
        for i in range(16)
    ]
    enc = MViT(
        spatial_size=(192, 192), temporal_size=96,
        block_setting=block_setting,
        residual_pool=True, residual_with_cls_embed=False,
        rel_pos_embed=True, proj_after_attn=True,
        stochastic_depth_prob=0.2, num_classes=400,
        patch_embed_kernel=(16, 2, 2), patch_embed_stride=(16, 2, 2),
        patch_embed_padding=(0, 0, 0),
    )
    enc.conv_proj = nn.Conv3d(1, 64, kernel_size=(16, 2, 2),
                              stride=(16, 2, 2), padding=(0, 0, 0))
    enc.head = nn.Identity()
    def hook_fn(module, input, output):
        _hook_storage['mvit_feat'] = output
    enc.norm.register_forward_hook(hook_fn)
    return enc


image_encoder = _build_encoder()


def extract_features(model, x):
    """Returns spatial features (B, C, 6, 12, 12) — channel-first."""
    if args.backbone == 'r3d18':
        out = model(x)
        return out.reshape(x.shape[0], config['latent_dim'], 6, 12, 12)

    elif args.backbone == 'swin3d_t':
        _ = model(x)
        feat = _hook_storage['swin_feat']                 # (B, 6, 12, 12, 768)
        return feat.permute(0, 4, 1, 2, 3)                # (B, 768, 6, 12, 12)

    else:  # mvit
        _ = model(x)
        tokens = _hook_storage['mvit_feat']                # (B, 1+N, C)
        spatial = tokens[:, 1:, :]                         # (B, N, C)
        B, N, C = spatial.shape
        return spatial.reshape(B, 6, 12, 12, C).permute(0, 4, 1, 2, 3)  # (B, C, 6, 12, 12)


# ── evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, classifier, dataloader, steps, device):
    model.eval()
    model.to(device)
    classifier.to(device)
    classifier.eval()

    pathologies = [ALL_PATHOLOGIES[i] for i in SELECTED_IDX]
    num_cls     = len(pathologies)
    patch_preds   = [[] for _ in range(num_cls)]
    patch_targets = [[] for _ in range(num_cls)]

    for batch in tqdm.tqdm(dataloader):
        inputs, _, labels, acc_no, annotations_volume = batch
        inputs = inputs.to(device)

        annotations_volume = (annotations_volume.reshape(
            annotations_volume.shape[0], 4, 6, 12, 12, 18).sum(dim=1) > 0).float()
        ann_np = annotations_volume.cpu().numpy()[..., SELECTED_IDX]
        B = ann_np.shape[0]

        if args.method == 'gradcam':
            with torch.enable_grad():
                feat_5d = extract_features(model, inputs)   # (B, C, 6, 12, 12)
                feat_5d.retain_grad()
                pooled = F.adaptive_avg_pool3d(feat_5d, 1).flatten(1)
                cls_scores = torch.sigmoid(classifier(pooled))  # (B, 18)

                for ci, global_ci in enumerate(SELECTED_IDX):
                    model.zero_grad()
                    classifier.zero_grad()
                    if feat_5d.grad is not None:
                        feat_5d.grad = None
                    cls_scores[:, global_ci].sum().backward(retain_graph=True)

                    grads = feat_5d.grad.clone()                        # (B, C, D, H, W)
                    weights = grads.mean(dim=(2, 3, 4), keepdim=True)   # (B, C, 1, 1, 1)
                    cam = (weights * feat_5d.detach()).sum(dim=1)        # (B, D, H, W)
                    cam = F.relu(cam)

                    cam_flat = cam.view(B, -1)
                    cam_min = cam_flat.min(dim=1, keepdim=True)[0]
                    cam_max = cam_flat.max(dim=1, keepdim=True)[0]
                    cam = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)
                    cam = cam.view(B, 6, 12, 12)
                    cam_np = cam.detach().cpu().numpy()

                    for b in range(B):
                        if ann_np[b, ..., ci].sum() == 0:
                            continue
                        patch_preds[ci].extend(cam_np[b].ravel())
                        patch_targets[ci].extend(ann_np[b, ..., ci].ravel())
        else:
            with torch.no_grad():
                feat_5d = extract_features(model, inputs)               # (B, C, 6, 12, 12)
                feat_hw = feat_5d.permute(0, 2, 3, 4, 1)               # (B, 6, 12, 12, C)
                output  = torch.sigmoid(classifier(feat_hw))            # (B, 6, 12, 12, 18)
                out_np  = output.detach().cpu().numpy()[..., SELECTED_IDX]
                for b in range(B):
                    for c in range(num_cls):
                        if ann_np[b, ..., c].sum() == 0:
                            continue
                        patch_preds[c].extend(out_np[b, ..., c].ravel())
                        patch_targets[c].extend(ann_np[b, ..., c].ravel())

    patch_dice, patch_prec, patch_rec, patch_spec, patch_acc = {}, {}, {}, {}, {}

    for c, name in enumerate(pathologies):
        probs = np.array(patch_preds[c])
        gts   = np.array(patch_targets[c]).astype(int)
        print(probs.shape, gts.shape)
        print(gts.sum())

        if gts.size == 0:
            patch_dice[name] = patch_prec[name] = patch_rec[name] = \
                patch_spec[name] = patch_acc[name] = np.nan
            continue

        best_f1, best_thr, best_bin = -1.0, 0.5, None
        for thr in np.linspace(0, 1, 101):
            bin_pred = (probs > thr).astype(int)
            f1 = f1_score(gts, bin_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr, best_bin = f1, thr, bin_pred

        tn, fp, fn, tp = confusion_matrix(gts, best_bin, labels=[0, 1]).ravel()
        patch_dice[name] = best_f1
        patch_prec[name] = tp / (tp + fp) if (tp + fp) else np.nan
        patch_rec[name]  = tp / (tp + fn) if (tp + fn) else np.nan
        patch_spec[name] = tn / (tn + fp) if (tn + fp) else np.nan
        patch_acc[name]  = (tp + tn) / (tp + tn + fp + fn)

        wandb.log({
            f"{name}_patch_threshold":      best_thr,
            f"{name}_patch_dice":           best_f1,
            f"{name}_patch_precision":      patch_prec[name],
            f"{name}_patch_recall":         patch_rec[name],
            f"{name}_patch_specificity":    patch_spec[name],
            f"{name}_patch_accuracy":       patch_acc[name],
            "global_step": steps,
        })

    wandb.log({
        "patch_avg_dice":        np.nanmean(list(patch_dice.values())),
        "patch_avg_precision":   np.nanmean(list(patch_prec.values())),
        "patch_avg_recall":      np.nanmean(list(patch_rec.values())),
        "patch_avg_specificity": np.nanmean(list(patch_spec.values())),
        "patch_avg_accuracy":    np.nanmean(list(patch_acc.values())),
    })


# ── training ─────────────────────────────────────────────────────────────────
def finetune():
    num_classes = 18
    classifier = nn.Linear(config['latent_dim'], num_classes).cuda()

    dataset_train_ = CTReportDatasetTrain(
        data_folder=data_train, csv_file=reports_file_train, labels=labels_train)
    dataset_valid_ = CTReportDatasetInfer(
        data_folder=data_valid, csv_file=reports_file_valid, labels=labels_valid)
    dataloader_train = DataLoader(
        dataset_train_, num_workers=4, batch_size=config['batchsize'], shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid_, num_workers=4, batch_size=1, shuffle=False)
    num_batches = len(dataloader_train)

    image_encoder.cuda()
    image_encoder.train()

    weights = torch.ones(len(SELECTED_IDX)).cuda()
    loss_fn = nn.BCELoss(weight=weights)

    optimizer = torch.optim.AdamW(
        list(image_encoder.parameters()) + list(classifier.parameters()),
        lr=config['lr'], weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'] * num_batches)

    for epoch in range(config['epochs']):
        for i, batch in tqdm.tqdm(enumerate(dataloader_train)):
            step = i + epoch * num_batches

            inputs, _, labels, _, annotations_volume = batch
            labels = labels.float().cuda()
            inputs = inputs.cuda()

            with torch.no_grad():
                labels = labels * (1 - config['smoothing']) + config['smoothing']

            feat_5d = extract_features(image_encoder, inputs)   # (B, C, 6, 12, 12)

            if args.method == 'noisyor':
                feat_hw = feat_5d.permute(0, 2, 3, 4, 1)                # (B, 6, 12, 12, C)
                patch_logits = torch.sigmoid(classifier(feat_hw))        # (B, 6, 12, 12, 18)
                cls_logits   = 1 - 0.005 * patch_logits
                cls_logits   = 1 - cls_logits.prod(dim=1).prod(dim=1).prod(dim=1)
            else:  # gradcam
                pooled     = F.adaptive_avg_pool3d(feat_5d, 1).flatten(1)  # (B, C)
                cls_logits = torch.sigmoid(classifier(pooled))             # (B, 18)

            loss = loss_fn(cls_logits[:, SELECTED_IDX], labels[:, SELECTED_IDX])

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad) for p in image_encoder.parameters()
                if p.grad is not None
            ]), 2)
            optimizer.step()
            scheduler.step()

            wandb.log({
                "train_cls_loss": loss.item(),
                "lr":             optimizer.param_groups[0]['lr'],
                "grad_norm":      grad_norm.item(),
            })

            if step != 0 and step % config['print_every'] == 0:
                evaluate_model(image_encoder, classifier, dataloader_valid,
                               steps=step, device='cuda')
                image_encoder.train()

            if step % config['save_every'] == 0:
                os.makedirs(config['save_dir'], exist_ok=True)
                model_to_save = (image_encoder.module
                                 if hasattr(image_encoder, 'module')
                                 else image_encoder)
                model_path = os.path.join(
                    config['save_dir'], f'checkpoint_{i}_epoch_{epoch + 1}.pt')
                print('Saving model to', model_path)
                torch.save({
                    'image_encoder': model_to_save.state_dict(),
                    'classifier':    classifier.state_dict(),
                }, model_path)


if __name__ == '__main__':
    finetune()
