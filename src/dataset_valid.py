import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from functools import partial
import torch.nn.functional as F
import tqdm
import monai.transforms as transforms
import random
import monai.data as data
import math, random, tqdm






class CTReportDatasetInfer(data.PersistentDataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()

        print(len(self.samples))
        random.shuffle(self.samples)

        self.pad_or_crop = transforms.ResizeWithPadOrCrop((120, 240, 240))
        self.random_crop = transforms.RandSpatialCrop(roi_size=(96, 192, 192), random_size=False)
        self.center_crop = transforms.CenterSpatialCrop(roi_size=(96, 192, 192))


        self.nii_to_tensor = partial(self.nii_img_to_tensor)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text

    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))
            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))
                for nii_file in nii_files:
                    file_name = os.path.basename(nii_file).replace("2.npz","1.npz")
                    case_id = file_name[:-4] if file_name.endswith(".npz") else file_name
                    annot_dir = os.path.join("annotations-valid", case_id)
                    if not os.path.isdir(annot_dir):
                        continue

                    accession_number = nii_file.split("/")[-1]
                    accession_number = accession_number.replace(".npz", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""

                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    
                    if len(onehotlabels) > 0:
                        samples.append((nii_file, text_final, onehotlabels[0]))
                        self.paths.append(nii_file)
        return samples


    def __len__(self):
        return len(self.samples)

    def get_high_res_annotation_mask(self, nii_file):
        fname = os.path.basename(nii_file).replace("2.npz", "1.npz")
        case_id = fname[:-4] if fname.endswith(".npz") else fname
        annot_dir = os.path.join("annotations-valid", case_id)
        if not os.path.isdir(annot_dir):
            return None
        class_mapping = {
            "coronary_wall_calcification": 1, "pericardial_effusion": 3, "arterial_wall_calcification": 4, "hiatal_hernia": 5, 
            "lymphadenopathy": 6, "atelectasis": 8, "lung_opacity": 10, 
            "pleural_effusion": 12, "consolidation": 15, "bronchiectasis": 16
        }
        vol = np.zeros((24, 12, 12, 18), dtype=np.float32)
        for npz_file in glob.glob(os.path.join(annot_dir, "*.npz")):
            stem = os.path.splitext(os.path.basename(npz_file))[0]
            disease_name = stem.split("__")[0]
            if disease_name in class_mapping:
                cid = class_mapping[disease_name]
                ann = np.load(npz_file)["arr_0"].astype(np.float32)
                vol[..., cid] = np.maximum(vol[..., cid], ann)
        vol = torch.from_numpy(vol.astype(np.float32))
        vol = vol.permute(3, 0, 1, 2)
        vol = vol.repeat_interleave(4,  dim=1)
        vol = vol.repeat_interleave(16, dim=2)
        vol = vol.repeat_interleave(16, dim=3)
        vol = self.pad_or_crop(vol)
        return vol

    def nii_img_to_tensor(self, path, annotations_mask):
        if annotations_mask is None:
            return None, None

        img = np.load(path)["arr_0"].astype(np.float32)
        hu_min, hu_max = -1000.0, 200.0
        img = np.clip(img * 1000.0, hu_min, hu_max)
        img = (img - hu_min) / (hu_max - hu_min)
        img = torch.from_numpy(img).unsqueeze(0)
        img = self.pad_or_crop(img)
        img = torch.rot90(img, k=-1, dims=(2, 3))
        img = torch.flip(img, dims=(3,))
        volume = torch.cat([img, annotations_mask], dim=0)
        volume = self.center_crop(volume)
        image_tensor = volume[0]
        mask_tensor  = volume[1:]
        
        annotations_grid = mask_tensor[:, 2::4, 8::16, 8::16].float()
   
        annotations_grid = annotations_grid.permute(1, 2, 3, 0).contiguous()
   
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor, annotations_grid

    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        high_res_mask = self.get_high_res_annotation_mask(nii_file)
        if high_res_mask is None:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        video_tensor, annotations_grid = self.nii_img_to_tensor(nii_file, high_res_mask)
        input_text = input_text.replace('"', '').replace('\'', '').replace('(', '').replace(')', '')
        name_acc = nii_file.split("/")[-2]
        return video_tensor, input_text, onehotlabels, name_acc, annotations_grid



if __name__ == "__main__":
    data_folder = "valid_preprocessed"
    csv_file = "dataset_radiology_text_reports_validation_reports.csv"
    labels = "dataset_multi_abnormality_labels_valid_predicted_labels.csv"

    dataset = CTReportDatasetInfer(data_folder, csv_file, labels=labels)
    print(len(dataset))
    for i in range(min(1000, len(dataset))):
        print(dataset[i][0].shape)

