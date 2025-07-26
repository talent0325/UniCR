from torchvision import models, transforms
import cv2
import numpy as np
import torch
from torchaudio.prototype.pipelines import VGGISH
import torchvision
import os
import glob
from tqdm import tqdm
import pickle
import pandas as pd
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop
from PIL import Image
import subprocess
import multiprocessing
from torchvision.transforms import functional as F
import parser
import argparse
import torchvision.transforms.functional as TF


class FakeLandmode(nn.Module):
    def __init__(self, base_img, target_ratio=(16, 9), resize_size=224, crop_size=224):
        """
        模拟横屏模式的预处理模块：
        1. 将输入图像在宽度方向填充为 target_ratio 的比例（例如16:9）；
        2. Resize 到 resize_size；
        3. Center crop 到 crop_size。

        Args:
            target_ratio: tuple, 目标宽高比，例如(16, 9)
            resize_size: int, resize 的尺寸（正方形）
            crop_size: int, 最终裁剪尺寸（正方形）
        """
        super(FakeLandmode, self).__init__()
        self.height, self.width  = base_img.shape[1], base_img.shape[2]

        self.target_ratio = target_ratio
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.resize_func = Resize((resize_size), interpolation=Image.BICUBIC)

    def forward(self, img):
        """
        Args:
            img: PIL Image，竖屏图像
        Returns:
            PIL Image，填充 + resize + center crop 后的图像
        """
        target_w = int(self.height * self.target_ratio[0] / self.target_ratio[1])
        pad_left = (target_w - self.width) // 2
        pad_right = target_w - self.width - pad_left

        # 填充宽度方向
        img = F.pad(img, padding=[pad_left, 0, pad_right, 0], fill=0)

        # Resize
        img = self.resize_func(img)

        # Center Crop
        img = F.center_crop(img, output_size=[self.crop_size, self.crop_size])

        return img

# VGG 是[512-640]，对应224窗口
# swin-transformer 是[440-512],对应192窗口
class RandomScaleCenterCrop(nn.Module):
    def __init__(self, base_img, min_short=440, max_short=512, window_size=192):
        super(RandomScaleCenterCrop, self).__init__()
        height, width  = base_img.shape[1], base_img.shape[2]
        short_side = min(width, height)
        # 随机选择一个缩放比例，使短边在[min_short, max_short]之间
        select_short_side = random.randint(min_short, max_short)
        # scale = select_short_side / short_side
        # new_width = int(width * scale)
        # new_height = int(height * scale)
        # self.resize_func = Resize((new_height, new_width), interpolation=Image.BICUBIC)
        self.resize_func = Resize(select_short_side, interpolation=Image.BICUBIC)
        self.crop_func = CenterCrop(window_size)
        pass

    def forward(self, img):
        img = self.resize_func(img)
        img = self.crop_func(img)
        return img


class Inception(nn.Module):
    def __init__(self, base_img, window_size):
        super(Inception, self).__init__()
        self.height, self.width  = base_img.shape[1], base_img.shape[2]
        self.whole_size = self.height * self.width
        self.window_size = window_size
        

    def forward(self, img):
        random_ratio = random.uniform(0.08, 1.0)
        target_pixel = int(self.whole_size * random_ratio)
        height_width_ratio = random.uniform(3/4, 4/3)
        target_height = int((target_pixel / height_width_ratio) ** 0.5)
        target_width = int(target_height * height_width_ratio)
        if target_height > self.height:
            target_height = self.height
        
        if target_width > self.width:
            target_width = self.width
        start_x = random.randint(0, self.width - target_width)
        start_y = random.randint(0, self.height - target_height)
        resize_func = Resize((self.window_size, self.window_size), interpolation=Image.BICUBIC)
        img = F.crop(img, top=start_y, left=start_x, height=target_height, width=target_width)
        img = resize_func(img)
        return img


def resize_and_pad(image, target_size=224):
    w, h = image.size
    # _, h, w = image.shape
    scale = target_size / max(w, h)  # 将长边缩放为 target_size
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    padding = (
        pad_w // 2, pad_h // 2,
        pad_w - pad_w // 2, pad_h - pad_h // 2
    )
    image = TF.pad(image, padding, fill=0, padding_mode='constant')
    return image
# class ScaleRandomCrop(nn.Module):
#     def __init__(self, base_img, window_size=192):
#         super(ScaleRandomCrop, self).__init__()
#         self.resize_func = Resize(window_size, interpolation=Image.BICUBIC)
#         self.crop_func = RandomCrop(window_size)

#     def forward(self, img):
#         img = self.resize_func(img)
#         img = self.crop_func(img)
#         return img

def ffmpeg_extraction(args):
    videofile, frame_dir = args
    basename = os.path.basename(videofile)[:-4]
    outdir = os.path.join(frame_dir, basename)
    os.makedirs(outdir, exist_ok=True)
    command = f"ffmpeg -i {videofile} -y -r 16 '{outdir}/%06d.jpg'"
    subprocess.call(command, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Extract features from Video frames with different preprocessing methods")
    parser.add_argument("--csv_path", type=str, default="dataset/csvfiles/meta_data_selected.csv", help="path to the csv file containing video ids")
    parser.add_argument("--frame_dir", type=str, default="dataset/data/selected_pm_frames/", help="path to the directory containing video frames")
    parser.add_argument("--video_root", type=str, default="dataset/data/videos/", help="path to the directory containing video files")
    parser.add_argument("--feature_root", type=str, default="dataset/feature/preprocess_visual_feature/", help="path to the directory to save extracted features")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--token", default="video_id", type=str, help="the column name of video id in the csv file")

    args = parser.parse_args()
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # # MARK: step1: Extract frames from the video file
    if not os.path.exists(args.frame_dir):
        os.makedirs(args.frame_dir)
    if not os.path.exists(args.feature_root):
        os.makedirs(args.feature_root)
        # 记录错误信息
    def log_error(vid, sec, error_msg):
        with open(error_log, "a") as f:
            f.write(f"Video: {vid}, Second: {sec}, Error: {error_msg}\n")
            
    error_log = os.path.join(args.feature_root, "error.txt")

    csv = pd.read_csv(args.csv_path)
    vids = csv[args.token].tolist()
    vids.sort()
    # 去除已存在特征的 vid
    existing_vids = set(os.path.splitext(f)[0] for f in os.listdir(args.feature_root) if f.endswith('.pkl'))
    vids = [vid for vid in vids if str(vid) not in existing_vids]
    with open(error_log, "a") as f:
        f.write(f"Total {len(vids)} videos to be processed.\n")
    mp4files = [os.path.join(args.video_root, str(vid) + ".mp4") for vid in vids]
    ffmpeg_args = [(mp4, args.frame_dir) for mp4 in mp4files]

    with multiprocessing.Pool(40) as p:
        p.map(ffmpeg_extraction, ffmpeg_args)

    # MARK: step2: Extract features from the frames
    device = torch.device("cuda")

    # 加载预训练的VGG19模型
    vgg19_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    vgg19_model = torch.nn.Sequential(*list(vgg19_model.children())[0])  # 移除分类层
    vgg19_model = vgg19_model.to(device)
    vgg19_model.eval()


    for vid in tqdm(vids):
        frames_path = os.path.join(args.frame_dir, str(vid))
        frames = glob.glob(frames_path + "/*.jpg")
        frames.sort()
        if len(frames) < 160:
            log_error(vid, 0, "frames less than 160")
            continue
        step = 16
        # 初始化预处理函数
        std_img = torchvision.io.read_image(frames[0])/255
        std_resize = transforms.Resize((224, 224), interpolation=Image.BICUBIC)

        scale_center_crop = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224)
        ])
        scale_rd_crop = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224)
        ])
        inception = Inception(std_img, 224)

        fakelandmode = FakeLandmode(std_img, target_ratio=(16, 9), resize_size=224, crop_size=224)

        normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        std_resize_features = []
        scale_center_features = []
        scale_rd_features = []
        inception_features = []
        longer_side_resize_features = []
        fakelandmode_features = []
        for sec in range(10):
            start = sec*step
            end = min((sec+1)*step, len(frames))
            sampled_idx = list(range(start, end))
            # sampled_idx = list(range(sec*step, (sec+1)*step))
            frames_selected = [frames[i] for i in sampled_idx]

            std_resize_frames_tensor = []
            scale_center_frames_tensor = []
            scale_rd_frames_tensor = []
            inception_frames_tensor = []
            longer_size_resize_frames_tensor = []
            fakelandmode_frames_tensor = []
            for frame_path in frames_selected:
                try:
                    img = Image.open(frame_path).convert("RGB")  # 读取图像并转换为 RGB 如果某段帧损坏跳过而不是终止整个进程
                except Exception as e:
                    log_error(vid, sec, f"frame {frame_path} error: {e}")
                    continue

                img_ori = img
                img = transforms.ToTensor()(img)  # 转换为张量
                to_tensor = transforms.ToTensor()

                try:
                    std_resize_frames_tensor.append(normalize(std_resize(img)))
                    scale_center_frames_tensor.append(normalize(scale_center_crop(img)))
                    scale_rd_frames_tensor.append(normalize(scale_rd_crop(img)))
                    inception_frames_tensor.append(normalize(inception(img)))
                    img_rp = resize_and_pad(img_ori, target_size=224)
                    longer_size_resize_frames_tensor.append(normalize(to_tensor(img_rp)))
                    fakelandmode_frames_tensor.append(normalize(fakelandmode(img)))
                except Exception as e:
                    log_error(vid, sec, f"transform error for frame {frame_path}: {e}")
                    continue
            
            try:
                std_resize_frames_tensor = torch.stack(std_resize_frames_tensor)
                scale_center_frames_tensor = torch.stack(scale_center_frames_tensor)
                scale_rd_frames_tensor = torch.stack(scale_rd_frames_tensor)
                inception_frames_tensor = torch.stack(inception_frames_tensor)
                longer_size_resize_frames_tensor = torch.stack(longer_size_resize_frames_tensor)
                fakelandmode_frames_tensor = torch.stack(fakelandmode_frames_tensor)
            except Exception as e:
                log_error(vid, sec, f"torch.stack error (maybe empty tensor list): {e}")
                continue  # 跳过该秒，继续处理下一个 sec
            # 提取特征
            with torch.no_grad():
                output = vgg19_model(std_resize_frames_tensor.to(device))
                output = output.permute(0, 2, 3, 1)
                output = torch.mean(output, dim=0)
                std_resize_features.append(output.squeeze())

                output = vgg19_model(scale_center_frames_tensor.to(device))
                output = output.permute(0, 2, 3, 1)
                output = torch.mean(output, dim=0)
                scale_center_features.append(output.squeeze())

                output = vgg19_model(scale_rd_frames_tensor.to(device))
                output = output.permute(0, 2, 3, 1)
                output = torch.mean(output, dim=0)
                scale_rd_features.append(output.squeeze())

                output = vgg19_model(inception_frames_tensor.to(device))
                output = output.permute(0, 2, 3, 1)
                output = torch.mean(output, dim=0)
                inception_features.append(output.squeeze())

                output = vgg19_model(longer_size_resize_frames_tensor.to(device))
                output = output.permute(0, 2, 3, 1)
                output = torch.mean(output, dim=0)
                longer_side_resize_features.append(output.squeeze())

                output = vgg19_model(fakelandmode_frames_tensor.to(device))
                output = output.permute(0, 2, 3, 1)
                output = torch.mean(output, dim=0)
                fakelandmode_features.append(output.squeeze())

        std_resize_features = torch.stack(std_resize_features)
        scale_center_features = torch.stack(scale_center_features)
        scale_rd_features = torch.stack(scale_rd_features)
        inception_features = torch.stack(inception_features)
        longer_side_resize_features = torch.stack(longer_side_resize_features)
        fakelandmode_features = torch.stack(fakelandmode_features)

        if std_resize_features.shape[0] != 10 \
                or scale_center_features.shape[0] != 10 \
                or scale_rd_features.shape[0] != 10 \
                or inception_features.shape[0] != 10 \
                or longer_side_resize_features.shape[0] != 10 \
                or fakelandmode_features.shape[0] != 10:
            log_error(vid, 0, "feature extraction error")
            continue
        # 保存特征
        feature_dict = {
            "std_resize": std_resize_features,
            "center_crop": scale_center_features,
            "random_crop": scale_rd_features,
            "inception": inception_features,
            "longer_side_resize": longer_side_resize_features,
            "fakelandmode": fakelandmode_features
        }

        save_path = os.path.join(args.feature_root, str(vid) + ".pkl")
        with open(save_path, "wb") as f:
            pickle.dump(feature_dict, f)

if __name__ == '__main__':
    main()