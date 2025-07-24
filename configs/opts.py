import argparse

def str2bool(str):
	return True if str.lower() == 'true' else False
parser = argparse.ArgumentParser(description="A project implemented in pyTorch")

# =========================== Learning Configs ============================

parser.add_argument('--n_epoch', type=int)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('--test_batch_size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--gpu', type=str)
parser.add_argument('--snapshot_pref', type=str)
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--clip_gradient', type=float)
parser.add_argument('--loss_weights', type=float)
parser.add_argument('--start_epoch', type=int)
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
parser.add_argument('--weight_decay', '--wd', type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

# =========================== Display Configs ============================
parser.add_argument('--print_freq', type=int)
parser.add_argument('--save_freq', type=int)
parser.add_argument('--eval_freq', type=int)

# =========================== Data Configs ============================
parser.add_argument('--ave', type=str2bool, default=False)
parser.add_argument('--avepm', type=str2bool, default=False)
parser.add_argument('--preprocess', type=str, default='None', help='std_resize, center_crop, random_crop, inception, longer_side_resize, None')
parser.add_argument('--audio_preprocess_mode', type=str, default='None', help="NMF1, NMF2, LMS, None")
parser.add_argument('--category_num', type=int, default=86)
parser.add_argument('--data_root', type=str, default='dataset/feature/features/')
parser.add_argument('--meta_root', type=str, default='dataset/csv/')
parser.add_argument('--v_feature_root', type=str, default='dataset/feature/preprocess_visual_feature')
parser.add_argument('--a_feature_root', type=str, default='dataset/feature/preprocess_audio_feature')

parser.add_argument('--logs_dir', type=str, default='logs/')
parser.add_argument('--warm_up', type=int, default=5)
parser.add_argument('--lr_cln', type=float, default=0.001)
parser.add_argument('--is_select', type=str2bool, default="True")
parser.add_argument('--time', type=str, default='')
parser.add_argument('--postscript', type=str, default='')

parser.add_argument('--use_cln', type=str2bool, default=True)
parser.add_argument('--use_saliency', type=str2bool, default=True)   
parser.add_argument('--seed', type=int, default=510)
parser.add_argument('--guide', type=str, default='audio', help="audio means only audio; co_guide means audio and visual guided")
parser.add_argument('--bgm', type=str, default='None', help="None, with, without")

