import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np


class AVEDataset(Dataset):
    def __init__(self, data_root, meta_root,split='train', ave=False, avepm=False, preprocess_mode='None', audio_process_mode="None", is_select=False, 
                 v_feature_root="dataset/feature/preprocess_visual_feature",
                 a_feature_root="dataset/feature/preprocess_audio_feature",
                 seed=43, 
                 bgm = "None"):
        super(AVEDataset, self).__init__()
        self.split = split
        assert not (ave and avepm), "enable two datsets at the same time"
        self.ave = ave
        self.avepm = avepm

        self.preprocess_mode = preprocess_mode
        self.audio_process_mode = audio_process_mode
        self.is_select = is_select
        self.a_feature_root = a_feature_root
        self.v_feature_root = v_feature_root
        self.data_root = data_root
        self.meta_root = meta_root
        self.seed = seed
        self.bgm = bgm

        if ave:
            self.meta_root = os.path.join(meta_root, 'ave')
        elif avepm:
            self.meta_root = os.path.join(meta_root, 'avepm')
        
        
        if self.audio_process_mode != 'None':   
            audio_processmeta = pd.read_csv(os.path.join(self.meta_root, "{}_meta.csv".format(audio_process_mode)))
            self.audio_process_ids = audio_processmeta["sample_id"].tolist()

        if self.is_select:
            self.meta_root = os.path.join(self.meta_root, 'select')
        
        if self.bgm == 'with':
            self.meta_root = os.path.join(self.meta_root, 'bgm')
        elif self.bgm == 'without':
            self.meta_root = os.path.join(self.meta_root, 'no_bgm')
        
        self.raw_gt = pd.read_csv(os.path.join(self.meta_root, "{}.csv".format(split)))
        # if self.bgm == 'with':
        #     self.raw_gt = pd.read_csv(os.path.join(self.meta_root, "{}_bgm.csv".format(split)))
        # elif self.bgm == 'without':
        #     self.raw_gt = pd.read_csv(os.path.join(self.meta_root, "{}_no_bgm.csv".format(split)))
        # elif self.bgm == 'None':
        #     self.raw_gt = pd.read_csv(os.path.join(self.meta_root, "{}.csv".format(split)))


    def __getitem__(self, index):
        sample_id = self.raw_gt.iloc[index]['sample_id']
        with open(os.path.join(self.data_root, f"{sample_id}.pkl"), 'rb') as f:
            data = pickle.load(f)
        video_features = data["video_features"]
        audio_features = data["audio_features"]
        labels = data["onehot_labels"]

        if self.preprocess_mode != 'None':
            with open(os.path.join(self.v_feature_root, f"{sample_id}.pkl"), 'rb') as vf:
                v_data= pickle.load(vf)
            video_features = v_data[self.preprocess_mode]
        
        if self.audio_process_mode != 'None':
            if sample_id in self.audio_process_ids and self.split == 'train':
                with open(os.path.join(self.a_feature_root, f"{self.audio_process_mode}", f"{sample_id}.pkl"), 'rb') as af:
                    audio_process_data = pickle.load(af)
                audio_features = audio_process_data["audio_features"]
        # if self.audio_process_mode != 'None':
        #     p = torch.rand(1).item()
        #     if sample_id in self.audio_process_ids and self.split == 'train' and p > 0.5:
        #         with open(os.path.join(self.a_feature_root, f"{self.audio_process_mode}", f"{sample_id}.pkl"), 'rb') as af:
        #             audio_process_data = pickle.load(af)
        #         audio_features = audio_process_data["audio_features"] 
        return video_features, audio_features, labels

    def __len__(self):
        return len(self.raw_gt)
    

class BGMDataset(Dataset):
    def __init__(self, data_root, meta_root,split='train', ave=False, avepm=False, preprocess_mode='None', audio_process_mode="None", is_select=False, 
                 v_feature_root="dataset/feature/preprocess_visual_feature",
                 a_feature_root="dataset/feature/preprocess_audio_feature",
                 seed=43):
        super(BGMDataset, self).__init__()
        self.split = split
        assert not (ave and avepm), "enable two datsets at the same time"
        self.ave = ave
        self.avepm = avepm

        self.preprocess_mode = preprocess_mode
        self.audio_process_mode = audio_process_mode
        self.is_select = is_select
        self.a_feature_root = a_feature_root
        self.v_feature_root = v_feature_root
        self.data_root = data_root
        self.meta_root = meta_root
        self.seed = seed
        
        
        if self.audio_process_mode != 'None':   
            audio_processmeta = pd.read_csv(os.path.join(self.meta_root, "{}_meta.csv".format(audio_process_mode)))
            self.audio_process_ids = audio_processmeta["sample_id"].tolist()
        
        self.raw_gt = pd.read_csv(os.path.join(self.meta_root, "{}.csv".format(split)))


    def __getitem__(self, index):
        sample_id = self.raw_gt.iloc[index]['sample_id']
        with open(os.path.join(self.data_root, f"{sample_id}.pkl"), 'rb') as f:
            data = pickle.load(f)
        video_features = data["video_features"]
        audio_features = data["audio_features"]
        labels = data["onehot_labels"]

        if self.preprocess_mode != 'None':
            with open(os.path.join(self.v_feature_root, f"{sample_id}.pkl"), 'rb') as vf:
                v_data= pickle.load(vf)
            video_features = v_data[self.preprocess_mode]
        
        if self.audio_process_mode != 'None':
            if sample_id in self.audio_process_ids and self.split == 'train':
                with open(os.path.join(self.a_feature_root, f"{self.audio_process_mode}", f"{sample_id}.pkl"), 'rb') as af:
                    audio_process_data = pickle.load(af)
                audio_features = audio_process_data["audio_features"]
        return video_features, audio_features, labels

    def __len__(self):
        return len(self.raw_gt)