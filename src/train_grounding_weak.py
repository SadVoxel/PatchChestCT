import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_valid import CTReportDatasetInfer
from dataset_train import CTReportDatasetTrain
import math
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, precision_score
import copy
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve


data_train = "train_preprocessed"
data_valid = "valid_preprocessed"
reports_file_train = "train_reports.csv"
reports_file_valid = "dataset_radiology_text_reports_validation_reports.csv"
labels_train = "dataset_multi_abnormality_labels_train_predicted_labels.csv"
labels_valid = "dataset_multi_abnormality_labels_valid_predicted_labels.csv"



import torchvision.models.video as models

config = {
    "lr": 1e-4,
    "wd": 1e-4,
    "warmup_length": 0, 
    "epochs": 20,
    "print_every": 500,
    "save_every": 500, 
    "batchsize": 10,
    "smoothing": 0.0,
    "dropout_prob": 0.0,
    "latent_dim": 512,
    "pretrained": True,
}

SELECTED_IDX = [1, 3, 4, 5, 6, 8, 10, 15, 16]

image_encoder = models.r3d_18(pretrained=config['pretrained'])
image_encoder.stem[0] = nn.Conv3d(1, 64, kernel_size=(2,2,2), stride=(2, 2, 2), padding=(0, 0, 0), bias=False)
image_encoder.avgpool = nn.Identity()
image_encoder.fc = nn.Identity()

config['save_dir'] = f""


wandb.init(project="", 
            name = config['save_dir'],
            config=config)


def find_threshold(probabilities, true_labels):
    """
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        probabilities (numpy.ndarray): Predicted probabilities.
        true_labels (numpy.ndarray): True labels.

    Returns:
        float: Optimal threshold.
    """
    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)
        confusion = confusion_matrix(true_labels, predictions)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        current_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if current_roc <= best_roc:
            best_roc = current_roc
            best_threshold = threshold

    return best_threshold

def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))

import torch.nn.functional as F

import numpy as np
import torch
import tqdm
from sklearn.metrics import f1_score, confusion_matrix


def evaluate_model(model, classifier, dataloader, steps, device):
    model.eval()
    model = model.to(device)
    classifier = classifier.to(device)
    classifier.eval()


    ALL_PATHOLOGIES = ['Medical material','Arterial wall calcification','Cardiomegaly',
                       'Pericardial effusion','Coronary artery wall calcification','Hiatal hernia',
                       'Lymphadenopathy','Emphysema','Atelectasis','Lung nodule','Lung opacity',
                       'Pulmonary fibrotic sequela','Pleural effusion','Mosaic attenuation pattern',
                       'Peribronchial thickening','Consolidation','Bronchiectasis',
                       'Interlobular septal thickening']
    pathologies = [ALL_PATHOLOGIES[i] for i in SELECTED_IDX]     
    num_cls = len(pathologies)

    patch_preds   = [[] for _ in range(num_cls)]   
    patch_targets = [[] for _ in range(num_cls)]   
    # ----------------------------------------------------------

    predictedall, realall, accs, df_rows = [], [], [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            inputs, _, labels, acc_no, annotations_volume = batch
            inputs  = inputs.to(device)
            labels  = labels.float().to(device)

            # forward
            img_feat = image_encoder(inputs)                         # [B,512,6,12,12]
            img_feat = img_feat.reshape(img_feat.shape[0], config['latent_dim'], 6,12,12)

            annotations_volume = (annotations_volume.reshape(annotations_volume.shape[0], 4, 6, 12, 12, 18).sum(dim=1) > 0).float()
            img_feat = img_feat.permute(0,2,3,4,1)   
            output   = torch.sigmoid(classifier(img_feat))           # [B,6,12,12,C]


            out_np = output.detach().cpu().numpy()[..., SELECTED_IDX]          # [B,6,12,12,9]
            ann_np = annotations_volume.detach().cpu().numpy()[..., SELECTED_IDX] #

            B = out_np.shape[0]
            for b in range(B):                                   
                for c in range(num_cls):
                    # 
                    if ann_np[b, ..., c].sum() == 0:             
                        continue
                    patch_preds[c].extend(out_np[b, ..., c].ravel())  
                    patch_targets[c].extend(ann_np[b, ..., c].ravel())
         
            realall.append(labels.detach().cpu().numpy()[0])
            save_out = output.cpu().numpy().mean(axis=(1, 2, 3))[:, SELECTED_IDX]
            predictedall.append(save_out[0])
            accs.append(acc_no[0])
            df_rows.append({"accession_name": acc_no[0],
                            **{pathologies[j]: float(save_out[0][j]) for j in range(num_cls)}})

    patch_dice, patch_prec, patch_rec, patch_spec, patch_acc = {}, {}, {}, {}, {}
    patch_best_thr = {}

    for c, name in enumerate(pathologies):
        probs = np.array(patch_preds[c])
        gts   = np.array(patch_targets[c]).astype(int)

        print(probs.shape, gts.shape)
        print(gts.sum())

        if gts.size == 0:                                       
            patch_best_thr[name] = np.nan
            patch_dice[name]     = np.nan
            patch_prec[name]     = np.nan
            patch_rec[name]      = np.nan
            patch_spec[name]     = np.nan
            patch_acc[name]      = np.nan
            continue


        best_f1, best_thr = -1.0, 0.5
        for thr in np.linspace(0, 1, 101):
            bin_pred = (probs > thr).astype(int)
            f1 = f1_score(gts, bin_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
                best_bin = bin_pred

        tn, fp, fn, tp = confusion_matrix(gts, best_bin, labels=[0,1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec  = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        acc  = (tp + tn) / (tp + tn + fp + fn)

        patch_best_thr[name] = best_thr
        patch_dice[name]     = best_f1
        patch_prec[name]     = prec
        patch_rec[name]      = rec
        patch_spec[name]     = spec
        patch_acc[name]      = acc


        wandb.log({f"{name}_patch_threshold": best_thr,
                   f"{name}_patch_dice": best_f1,
                   f"{name}_patch_dice_histogram": wandb.Histogram(best_f1),
                   f"{name}_patch_precision": prec,
                   f"{name}_patch_recall": rec,
                   f"{name}_patch_specificity": spec,
                   f"{name}_patch_accuracy": acc,
                   "global_step": steps})

    wandb.log({"patch_avg_dice":        np.nanmean(list(patch_dice.values())),
               "patch_avg_precision":   np.nanmean(list(patch_prec.values())),
               "patch_avg_recall":      np.nanmean(list(patch_rec.values())),
               "patch_avg_specificity": np.nanmean(list(patch_spec.values())),
               "patch_avg_accuracy":    np.nanmean(list(patch_acc.values()))})



import torch
import torch.nn.functional as F
from typing import Union, Tuple



def finetune():
    num_classes = 18
    classifier = nn.Linear(config['latent_dim'], num_classes).cuda()

    dataset_train = CTReportDatasetTrain(data_folder=data_train, csv_file=reports_file_train, labels = labels_train)
    dataset_valid = CTReportDatasetInfer(data_folder=data_valid, csv_file=reports_file_valid, labels = labels_valid)
    dataloader_train = DataLoader(dataset_train, num_workers=4, batch_size=config['batchsize'], shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, num_workers=4, batch_size=1, shuffle=False)
    num_batches = len(dataloader_train)

    image_encoder.cuda()
    image_encoder.train()
    image_encoder.to("cuda")


    weights = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).cuda()
    weights = weights[SELECTED_IDX]
    loss_fn = torch.nn.BCELoss(weight=weights)
    optimizer = torch.optim.AdamW(list(image_encoder.parameters()) + list(classifier.parameters()), lr=config['lr'], weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] * len(dataloader_train))


    for epoch in range(config['epochs']):
        for i, batch in tqdm.tqdm(enumerate(dataloader_train)):
            step = i + epoch * num_batches

            inputs, _, labels, _, annotations_volume = batch
            labels = labels.float().cuda()
            inputs = inputs.cuda()
            annotations_volume = annotations_volume.cuda()

            with torch.no_grad():
                labels = labels * (1-config['smoothing']) + config['smoothing']
            
            image_features = image_encoder(inputs)
            image_features = image_features.reshape(image_features.shape[0], config['latent_dim'], 6, 12, 12)
            image_features = image_features.permute(0,2,3,4,1)
            patch_logits = classifier(image_features)
            patch_logits = torch.sigmoid(patch_logits)

            cls_logits = 1 - 0.005 * patch_logits
            cls_logits = torch.prod(cls_logits, dim=1)
            cls_logits = torch.prod(cls_logits, dim=1)
            cls_logits = torch.prod(cls_logits, dim=1)
            cls_logits = 1 - cls_logits

   
            loss_cls = loss_fn(cls_logits[:,SELECTED_IDX], labels[:,SELECTED_IDX])
            loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
      
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in image_encoder.parameters() if p.grad is not None]), 2)

            optimizer.step()
            scheduler.step()

            wandb.log({
                "train_cls_loss": loss_cls.item(), 
                "lr": optimizer.param_groups[0]['lr'], 
                "grad_norm": grad_norm.item(),
            })

            if step != 0 and step % config['print_every'] == 0:
         
                evaluate_model(image_encoder, classifier, dataloader_valid, steps=step, device="cuda")
                image_encoder.train()

            if step % config['save_every'] == 0:
                os.makedirs(config['save_dir'], exist_ok=True)

                model_to_save = image_encoder.module if hasattr(image_encoder, 'module') else image_encoder

                model_path = os.path.join(config['save_dir'], f'checkpoint_{i}_epoch_{epoch+1}.pt')
                print('Saving model to', model_path)
                final_model = {
                    'image_encoder': model_to_save.state_dict(),
                    'classifier': classifier.state_dict(),
                }
                torch.save(final_model, model_path)


if __name__ == '__main__':

    finetune()
