import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets_helper import *
from .main_model import *

class CS_Model(nn.Module):
    def __init__(self, config, args):
        super(CS_Model, self).__init__()
        self.config = config
        self.args = args
        self.d_model = self.config['d_model']
        self.categoty_num = self.config['category_num']
        dh = int(self.d_model / 2)
        self.use_cln = args.use_cln
        self.use_saliency = args.use_saliency

        self.saliency = PerFrameSaliency(in_channels=512)

        self.cln_a = CLN(d=self.d_model, dh=dh)
        self.cln_v = CLN(d=self.d_model, dh=dh)
        self.cln_m = CLN(d=self.d_model, dh=dh)
        self.representation = supv_main_model(config)

        self.predictor_a = nn.Linear(self.d_model, self.categoty_num)
        self.predictor_v = nn.Linear(self.d_model, self.categoty_num)
        self.predictor_m = nn.Linear(self.d_model, self.categoty_num)

        self.localize_a = SupvLocalizeModule(self.d_model, self.categoty_num)
        self.localize_v = SupvLocalizeModule(self.d_model, self.categoty_num)
        self.localize_m = SupvLocalizeModule(self.d_model, self.categoty_num)

    def forward(self, audio_feature, video_feature):
        if self.use_saliency:
            video_feature, sal_map = self.saliency(video_feature)

        f_a, f_v, f_m = self.representation(audio_feature, video_feature)

        is_event_score_a, p_a = self.localize_a(f_a)
        is_event_score_v, p_v = self.localize_v(f_v)
        is_event_score_m, p_m = self.localize_m(f_m)

        c_a = self.cln_a(f_a)
        c_v = self.cln_v(f_v)
        c_m = self.cln_m(f_m)
        
        if self.use_saliency:
            return is_event_score_a, p_a, is_event_score_v, p_v, is_event_score_m, p_m, c_a, c_v, c_m, sal_map
        return is_event_score_a, p_a, is_event_score_v, p_v, is_event_score_m, p_m, c_a, c_v, c_m
 