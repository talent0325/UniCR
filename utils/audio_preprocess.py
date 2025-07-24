import librosa
import numpy as np
import torch
import torchaudio
from sklearn.decomposition import NMF
from pathlib import Path
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.decomposition import NMF
from pathlib import Path
from torchvision import models, transforms
import cv2
import numpy as np
import torch
import numpy as np
from torchaudio.prototype.pipelines import VGGISH
import torchaudio
import os
import glob
from tqdm import tqdm
import pickle
import torch
from torchvision import transforms, models
from PIL import Image
import torchvision.transforms.functional as TF

def get_BGM_template(video_id: str, video_root_dir: str, event_start: float, event_end: float, video_duration: float = None) -> torch.Tensor:
    """
    提取视频中适合作为背景音乐模板的音频片段。
    
    Args:
        video_id (str): 视频ID。
        video_root_dir (str): 视频根目录。
        event_start (float): 目标事件开始时间（秒）。
        event_end (float): 目标事件结束时间（秒）。
        video_duration (float, optional): 视频总时长，如果为None则从文件中读取。
    
    Returns:
        torch.Tensor: 提取的背景音乐模板波形。
    """
    # 如果未提供视频时长，则从文件中读取
    video_path = Path(video_root_dir) / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 使用torchaudio获取音频信息
    metadata = torchaudio.info(str(video_path))
    if video_duration is None:
        video_duration = metadata.num_frames / metadata.sample_rate
    
    # 计算事件持续时间
    event_duration = event_end - event_start
    
    # 寻找最长的非事件区间
    pre_event_duration = event_start
    post_event_duration = video_duration - event_end
    
    # 理想的模板长度（秒）
    ideal_template_duration = min(5.0, event_duration)  # 不超过事件本身的长度
    
    if pre_event_duration >= post_event_duration and pre_event_duration > 0:
        # 从事件前提取
        template_start = max(0, event_start - ideal_template_duration)
        template_end = event_start
    elif post_event_duration > 0:
        # 从事件后提取
        template_start = event_end
        template_end = min(video_duration, event_end + ideal_template_duration)
    else:
        # 如果前后都没有足够空间，则取视频开头（不理想但是备选方案）
        template_start = 0
        template_end = min(ideal_template_duration, event_start * 0.5)
    
    # 加载对应时间段的音频
    # waveform, sample_rate = torchaudio.load(
    #     str(video_path),
    #     frame_offset=int(template_start * metadata.sample_rate),
    #     num_frames=int((template_end - template_start) * metadata.sample_rate)
    # )
    num_frames = int((template_end - template_start) * metadata.sample_rate)
    if num_frames <= 0:
        print(f"[警告] 无法从 {video_id} 中提取有效背景模板：时间段无效 ({template_start:.2f} ~ {template_end:.2f})")
        # return torch.zeros(1, int(metadata.sample_rate * 1.0))  # 返回1秒的静音
        return None

    waveform, sample_rate = torchaudio.load(
        str(video_path),
        frame_offset=int(template_start * metadata.sample_rate),
        num_frames=num_frames
    )

    
    return waveform

def nmf_BGM_removal(waveform: torch.Tensor, bgm_waveform: torch.Tensor, n_fft=2048, hop_length=512) -> torch.Tensor:
    """
    非负矩阵分解（NMF）用于分离背景音乐和目标事件。
    
    Args:
        waveform (torch.Tensor): 输入音频波形，包含事件和BGM。
        bgm_waveform (torch.Tensor): 仅包含背景音乐的波形。
    
    Returns:
        torch.Tensor: 分离出的目标事件音频。
    """
    # 检查输入是否为None
    if bgm_waveform is None:
        print("警告: BGM模板为None，返回原始波形")
        return waveform
    
    # 转换为numpy数组
    waveform_np = waveform.numpy()
    bgm_waveform_np = bgm_waveform.numpy()
    
    # 计算STFT参数
    n_fft = n_fft
    hop_length = hop_length
    
    # 对输入音频计算STFT
    if waveform_np.shape[0] > 1:  # 如果是立体声，取平均
        waveform_mono = np.mean(waveform_np, axis=0)
    else:
        waveform_mono = waveform_np[0]
    
    # 对BGM模板计算STFT
    if bgm_waveform_np.shape[0] > 1:  # 如果是立体声，取平均
        bgm_mono = np.mean(bgm_waveform_np, axis=0)
    else:
        bgm_mono = bgm_waveform_np[0]
    
    # 检查并处理bgm_waveform长度为0的情况
    if len(bgm_mono) == 0:
        print("警告: BGM模板长度为0，返回原始波形")
        return waveform
    
    # 处理bgm_waveform和waveform长度不一致的情况
    if len(bgm_mono) < len(waveform_mono):
        # 如果BGM模板较短，则循环重复直到长度足够
        repeats = int(np.ceil(len(waveform_mono) / len(bgm_mono)))
        bgm_mono = np.tile(bgm_mono, repeats)[:len(waveform_mono)]
    elif len(bgm_mono) > len(waveform_mono):
        # 如果BGM模板较长，则截断
        bgm_mono = bgm_mono[:len(waveform_mono)]
    
    # 计算幅度谱
    D_waveform = np.abs(librosa.stft(waveform_mono, n_fft=n_fft, hop_length=hop_length))
    D_bgm = np.abs(librosa.stft(bgm_mono, n_fft=n_fft, hop_length=hop_length))
    
    # 使用NMF分解
    # 初始化BGM模板作为一个固定的基向量
    W_init = np.zeros((D_waveform.shape[0], 2), dtype=D_waveform.dtype)
    W_init[:, 0] = np.mean(D_bgm, axis=1)  # 第一个基向量是BGM模板
    H_init = np.abs(np.random.rand(2, D_waveform.shape[1])).astype(D_waveform.dtype) + 0.1

    if np.all(W_init == 0):
        print("W_init 全为 0，跳过 NMF")
        return waveform  # 或其他备选方案

    # 应用NMF
    model = NMF(n_components=2, init='custom', random_state=0)
    
    W = model.fit_transform(D_waveform, W=W_init, H=H_init)
    H = model.components_
    
    # 重构BGM和事件声音
    D_bgm_reconstructed = np.outer(W[:, 0], H[0, :])
    D_event_reconstructed = np.outer(W[:, 1], H[1, :])
    
    # 使用维纳滤波器分离
    mask_event = D_event_reconstructed / (D_bgm_reconstructed + D_event_reconstructed + 1e-10)
    
    # 应用掩码到原始STFT
    D_waveform_complex = librosa.stft(waveform_mono, n_fft=n_fft, hop_length=hop_length)
    D_event_complex = D_waveform_complex * mask_event
    
    # 逆STFT重建时域信号
    event_signal = librosa.istft(D_event_complex, hop_length=hop_length)
    
    # 转换回torch张量
    event_tensor = torch.from_numpy(event_signal).float().unsqueeze(0)
    
    return event_tensor
# 3.2 非负矩阵分解（保留目标事件）
# 使用预先计算好的事件模板库进行NMF分离


def load_event_templates_and_find_best(waveform: torch.Tensor, sample_rate: int, category: int, template_path: str, n_fft=2048, hop_length=512) -> torch.Tensor:
    """
    加载事件模板库并为给定样本找到最匹配的模板
    
    Args:
        waveform (torch.Tensor): 输入音频波形
        sample_rate (int): 采样率
        category (int): 目标事件类别
        template_path (str): 模板库文件路径
    
    Returns:
        torch.Tensor: 最匹配的模板音频
    """
    # 加载事件模板库
    with open(template_path, 'rb') as f:
        templates = pickle.load(f)
    
    if category not in templates:
        raise ValueError(f"类别 {category} 在模板库中不存在")
    
    category_templates = templates[category]
    
    # 确保输入是单声道
    if waveform.shape[0] > 1:
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
    else:
        waveform_mono = waveform
    
    # 计算输入音频的特征
    n_fft = n_fft
    hop_length = hop_length
    waveform_stft = torch.stft(
        waveform_mono.squeeze(0), 
        n_fft=n_fft, 
        hop_length=hop_length, 
        window=torch.hann_window(n_fft), 
        return_complex=True
    )
    waveform_mag = torch.abs(waveform_stft).numpy()
    
    # 计算每个模板与输入音频的相似度
    best_similarity = -float('inf')
    best_template = None
    
    for cluster_id, template_dict in category_templates.items():
        template_audio = template_dict['audio']
        template_sr = template_dict['sample_rate']
        
        # 重采样模板音频到与输入音频相同的采样率
        if template_sr != sample_rate:
            template_audio = torchaudio.functional.resample(template_audio, template_sr, sample_rate)
        
        # 确保模板是单声道
        if template_audio.shape[0] > 1:
            template_audio = torch.mean(template_audio, dim=0, keepdim=True)
        
        # 计算模板的STFT
        template_stft = torch.stft(
            template_audio.squeeze(0), 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=torch.hann_window(n_fft), 
            return_complex=True
        )
        template_mag = torch.abs(template_stft).numpy()
        
        # 如果模板时间长度大于输入，截断模板
        if template_mag.shape[1] > waveform_mag.shape[1]:
            template_mag = template_mag[:, :waveform_mag.shape[1]]
        
        # 计算相似度（使用频谱相关性）
        # 将模板调整为与输入相同的时间长度
        if template_mag.shape[1] < waveform_mag.shape[1]:
            # 通过重复模板来匹配输入长度
            repeats = int(np.ceil(waveform_mag.shape[1] / template_mag.shape[1]))
            template_mag_extended = np.tile(template_mag, (1, repeats))
            template_mag = template_mag_extended[:, :waveform_mag.shape[1]]
        
        # 计算频谱相关性
        template_flat = template_mag.flatten()
        waveform_flat = waveform_mag.flatten()
        
        # 归一化
        template_norm = template_flat / (np.linalg.norm(template_flat) + 1e-8)
        waveform_norm = waveform_flat / (np.linalg.norm(waveform_flat) + 1e-8)
        
        # 计算余弦相似度
        similarity = np.dot(template_norm, waveform_norm)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_template = template_audio
    
    return best_template

def nmf_event_preservation(waveform: torch.Tensor, target_event_template: torch.Tensor, n_fft=2048, hop_length=512, threshold1=0.3, threshold2=0.1, threshold3=0.7) -> torch.Tensor:
    """
    使用NMF和目标事件模板分离目标事件，保留与模板相似的音频部分
    
    Args:
        waveform (torch.Tensor): 输入音频波形
        target_event_template (torch.Tensor): 目标事件模板
    
    Returns:
        torch.Tensor: 保留的目标事件音频
    """
    # 检查输入是否为None
    if target_event_template is None:
        print("警告: 事件模板为None，返回原始波形")
        return waveform
    
    # 转换为numpy数组
    waveform_np = waveform.numpy()
    event_template_np = target_event_template.numpy()
    
    # 计算STFT参数
    n_fft = n_fft
    hop_length = hop_length
    
    # 对输入音频计算STFT
    if waveform_np.shape[0] > 1:  # 如果是立体声，取平均
        waveform_mono = np.mean(waveform_np, axis=0)
    else:
        waveform_mono = waveform_np[0]
    
    # 对事件模板计算STFT
    if event_template_np.shape[0] > 1:  # 如果是立体声，取平均
        event_mono = np.mean(event_template_np, axis=0)
    else:
        event_mono = event_template_np[0]
    
    # 检查并处理事件模板长度为0的情况
    if len(event_mono) == 0:
        print("警告: 事件模板长度为0，返回原始波形")
        return waveform
    
    # 处理事件模板和输入音频长度不一致的情况
    if len(event_mono) < len(waveform_mono):
        # 如果事件模板较短，则循环重复直到长度足够
        repeats = int(np.ceil(len(waveform_mono) / len(event_mono)))
        event_mono = np.tile(event_mono, repeats)[:len(waveform_mono)]
    elif len(event_mono) > len(waveform_mono):
        # 如果事件模板较长，则截断
        event_mono = event_mono[:len(waveform_mono)]
    
    # 计算幅度谱
    D_waveform = np.abs(librosa.stft(waveform_mono, n_fft=n_fft, hop_length=hop_length))
    D_event = np.abs(librosa.stft(event_mono, n_fft=n_fft, hop_length=hop_length))
    
    # 使用NMF分解
    # 初始化事件模板作为一个固定的基向量
    W_init = np.zeros((D_waveform.shape[0], 2), dtype=D_waveform.dtype)
    W_init[:, 0] = np.mean(D_event, axis=1)  # 第一个基向量是事件模板
    H_init = np.abs(np.random.rand(2, D_waveform.shape[1])).astype(D_waveform.dtype) + 0.1

    # 应用NMF
    model = NMF(n_components=2, init='custom', random_state=0)
    
    W = model.fit_transform(D_waveform, W=W_init, H=H_init)
    H = model.components_
    
    # 重构事件和背景声音
    D_event_reconstructed = np.outer(W[:, 0], H[0, :])
    D_background_reconstructed = np.outer(W[:, 1], H[1, :])
    
    # 计算事件模板与分解结果的相似度
    template_similarity = np.corrcoef(W[:, 0], np.mean(D_event, axis=1))[0, 1]
    
    # 根据相似度调整掩码
    if template_similarity > 0.5:  # 如果相似度高，保留更多事件成分
        mask_event = D_event_reconstructed / (D_event_reconstructed + D_background_reconstructed + 1e-10)
        mask_event = np.clip(mask_event, threshold1, 1.0)  # 确保至少保留30%的事件成分
    else:  # 如果相似度低，使用更保守的掩码
        mask_event = D_event_reconstructed / (D_event_reconstructed + D_background_reconstructed + 1e-10)
        mask_event = np.clip(mask_event, threshold2, threshold3)  # 保留10%-70%的事件成分
    
    # 应用掩码到原始STFT
    D_waveform_complex = librosa.stft(waveform_mono, n_fft=n_fft, hop_length=hop_length)
    D_event_complex = D_waveform_complex * mask_event
    
    # 逆STFT重建时域信号
    event_signal = librosa.istft(D_event_complex, hop_length=hop_length)
    
    # 转换回torch张量
    event_tensor = torch.from_numpy(event_signal).float().unsqueeze(0)
    
    return event_tensor 


# 4. LMS滤波
# 每个类别都有3个模板样本。以模板样本作为参考信号，从带BGM的样本中计算并学习适用于该类别的滤波器组，从而将BGM从样本中去除。

def lms_filter(waveform: torch.Tensor, template: torch.Tensor, filter_length=64, step_size=0.01, iterations=1000) -> torch.Tensor:
    """
    使用LMS（最小均方）自适应滤波器从音频中去除背景音乐
    
    参数:
        waveform: 输入音频波形，包含事件和背景音乐
        template: 模板音频波形，作为参考信号
        filter_length: 滤波器长度
        step_size: 学习率
        iterations: 迭代次数
        
    返回:
        去除背景音乐后的音频波形
    """
    # 确保输入是单声道
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if template.dim() > 1 and template.shape[0] > 1:
        template = template.mean(dim=0, keepdim=True)
    
    # 将张量转换为一维数组
    x = waveform.squeeze().numpy()  # 输入信号
    d = template.squeeze().numpy()  # 期望信号（模板）
    
    # 确保模板长度适合处理
    if len(d) > len(x):
        d = d[:len(x)]
    elif len(d) < len(x):
        # 通过重复模板来匹配输入长度
        repeats = int(np.ceil(len(x) / len(d)))
        d_extended = np.tile(d, repeats)
        d = d_extended[:len(x)]
    
    # 初始化滤波器系数
    w = np.zeros(filter_length)
    
    # 初始化输出信号
    y = np.zeros_like(x)
    
    # LMS算法实现
    for i in range(filter_length, len(x)):
        # 提取当前输入窗口
        x_window = x[i-filter_length:i]
        
        # 计算滤波器输出
        y[i] = np.dot(w, x_window)
        
        # 计算误差
        e = d[i] - y[i]
        
        # 更新滤波器系数
        w = w + step_size * e * x_window
    
    # 将结果转换回PyTorch张量
    filtered_signal = torch.from_numpy(y).unsqueeze(0).float()
    
    return filtered_signal

def load_templates(category, template_metadata_path='template/templates/template_metadata.csv', videos_root_dir='/data1/datasets/PM-400/data/data'):
    """
    加载指定类别的所有模板音频
    
    参数:
        category: 事件类别ID
        template_metadata_path: 模板元数据CSV文件路径
        videos_root_dir: 视频文件根目录
        
    返回:
        模板音频列表
    """
    # 读取模板元数据
    template_df = pd.read_csv(template_metadata_path)
    
    # 筛选指定类别的模板
    category_templates = template_df[template_df['category'] == category]
    
    templates = []
    for _, row in category_templates.iterrows():
        video_id = row['video_id']
        start_time = row['event_start']
        end_time = row['event_end']
        
        # 加载视频音频
        video_path = f"{videos_root_dir}/{video_id}.mp4"
        if os.path.exists(video_path):
            waveform, sr = torchaudio.load(video_path)
            
            # 提取事件部分
            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            
            if end_frame <= waveform.shape[1]:
                event_audio = waveform[:, start_frame:end_frame]
                templates.append(event_audio)
    
    return templates

import parser
import os
from tqdm import tqdm
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess audio data')
    parser.add_argument('--annotation_file', type=str, default='dataset/csvfiles/annotations.csv', help='annotation file path')
    parser.add_argument('--preprocess_type', type=str, default='NMF1', help='preprocess type:NMF1/NMF2/LMS')
    parser.add_argument('--template_path', type=str, default='dataset/templates.pkl', help='template path')
    parser.add_argument('--output_root', type=str, default='dataset/data/processed_audios/', help='output path root')
    parser.add_argument('--videos_root_dir', type=str, default='dataset/data/videos', help='videos root directory')
    parser.add_argument('--n_fft', type=int, default=2048, help='STFT parameter')
    parser.add_argument('--hop_length', type=int, default=512, help='STFT parameter')
    parser.add_argument('--LMS_filter_length', type=int, default=128, help='LMS filter length')
    parser.add_argument('--LMS_step_size', type=float, default=0.01, help='LMS step size')
    parser.add_argument('--LMS_iterations', type=int, default=1000, help='LMS iterations')
    parser.add_argument('--threshold1', type=float, default=0.3, help='NMF event preservation threshold')
    parser.add_argument('--threshold2', type=float, default=0.1, help='NMF event preservation threshold')
    parser.add_argument('--threshold3', type=float, default=0.7, help='NMF event preservation threshold')

    parser.add_argument('--feature_root', type=str, default='dataset/feature/preprocess_audio_feature', help='feature root directory')

    args = parser.parse_args()
    annotation_file = args.annotation_file
    preprocess_type = args.preprocess_type
    template_path = args.template_path
    output_root = args.output_root
    videos_root_dir = args.videos_root_dir
    n_fft = args.n_fft
    hop_length = args.hop_length
    LMS_filter_length = args.LMS_filter_length
    LMS_step_size = args.LMS_step_size
    LMS_iterations = args.LMS_iterations
    threshold1 = args.threshold1
    threshold2 = args.threshold2
    threshold3 = args.threshold3



    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_dir = os.path.join(output_root, preprocess_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    error_log = os.path.join(output_dir, 'error.log')
    
    df = pd.read_csv(annotation_file)
    # 定义音乐类别
    music_classes = [0, 1, 3, 12, 18, 26, 27]
    # music_classes = [4, 5, 10, 81, 84, 86, 92, 93, , 102, 106, 108, 118, 120, 196, 346, 347, 348, 349, 352]
    # filtered_samples = df[(df['haveBGM'] == 1) & 
    #                     (df['audio_irrelevant'] == 0) & 
    #                     (~df['label'].isin(music_classes))]
    filtered_samples = df[ (~df['label'].isin(music_classes))]
    print(f"total {len(filtered_samples)} samples")
    
    # for idx in tqdm(range(len(filtered_samples)), desc="Audio preprocess"):
    #     sample = filtered_samples.iloc[idx]
    #     onset = sample['onset']
    #     offset = sample['offset']
    #     video_id = sample['video_id']
    #     # video_event_start = sample['event_start']
    #     # video_event_end = sample['event_end']
    #     video_event_start = sample['onset']
    #     video_event_end = sample['offset']
    #     # video_duration = sample['duration']
    #     video_duration = sample['offset'] - sample['onset']
    #     # sample_id = sample['sample_id']
    #     sample_id = sample['video_id']
    #     category = sample['label']
    #     output_path = os.path.join(output_dir, f"{sample_id}.wav")
    #     if os.path.exists(output_path):
    #         continue

    #     video_path = os.path.join(videos_root_dir, f"{video_id}.mp4")
    #     if not os.path.exists(video_path):
    #         with open(error_log, "a") as f:
    #             f.write(f"[{sample_id}] {video_path} does not exist.\n")
    #         continue
    #     try:
    #         waveform, sr = torchaudio.load(video_path)
    #     except Exception as e:
    #         with open(error_log, "a") as f:
    #             f.write(f"[{sample_id}] Error loading audio: {e}\n")
    #         continue

    #     if preprocess_type == 'NMF1':
    #         try:
    #             bgm_waveform = get_BGM_template(
    #                 video_id=video_id,
    #                 video_root_dir=videos_root_dir,
    #                 event_start=video_event_start,
    #                 event_end=video_event_end,
    #                 video_duration=video_duration
    #             )
    #         except Exception as e:
    #             with open(error_log, "a") as f:
    #                 f.write(f"[{sample_id}] Error extracting BGM template: {e}\n")
    #             continue

    #         try:
    #             filtered_audio = nmf_BGM_removal(waveform, bgm_waveform, n_fft=n_fft, hop_length=hop_length)
    #         except Exception as e:
    #             with open(error_log, "a") as f:
    #                 f.write(f"[{sample_id}] Error in NMF BGM removal: {e}\n")
    #             continue
    #     elif preprocess_type == 'NMF2':
    #         try:
    #             best_template = load_event_templates_and_find_best(waveform, sr, category, template_path, n_fft=n_fft, hop_length=hop_length)
    #         except Exception as e:
    #             with open(error_log, "a") as f:
    #                 f.write(f"[{sample_id}] Error finding best template: {e}\n")
    #             continue

    #         # 使用NMF分离目标事件
    #         try:
    #             filtered_audio = nmf_event_preservation(waveform, best_template, n_fft=n_fft, hop_length=hop_length, threshold1=threshold1, threshold2=threshold2, threshold3=threshold3)
    #         except Exception as e:
    #             with open(error_log, "a") as f:
    #                 f.write(f"[{sample_id}] Error extracting event: {e}\n")
    #             continue
    #     elif preprocess_type == 'LMS':
    #         # 选择事件区域
    #         start_frame = int(onset * sr)
    #         end_frame = int(offset * sr)
    #         waveform_event = waveform[:, start_frame:end_frame]
    #         # 加载类别模板
    #         try:
    #             best_template = load_event_templates_and_find_best(waveform, sr, category, template_path, n_fft=n_fft, hop_length=hop_length)
    #         except Exception as e:
    #             with open(error_log, "a") as f:
    #                 f.write(f"[{sample_id}] Error finding best template: {e}\n")
    #             continue

    #         # 应用LMS滤波
    #         try:
    #             filtered_audio = lms_filter(waveform_event, best_template, filter_length=LMS_filter_length, step_size=LMS_step_size, iterations=LMS_iterations)
    #         except Exception as e:
    #             with open(error_log, "a") as f:
    #                 f.write(f"[{sample_id}] Error applying LMS filter: {e}\n")
    #             continue
        
    #     # 保存结果
    #     try:
    #         torchaudio.save(output_path, filtered_audio, sr)
    #     except Exception as e:
    #         with open(error_log, "a") as f:
    #             f.write(f"[{sample_id}] Error saving audio: {e}\n")
    # MARK: get feature
    # 获取VGGish特征
    feature_path = os.path.join(args.feature_root, preprocess_type)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    error_log_feature = os.path.join(feature_path, 'error.log')
    # 过滤掉已经提取过特征的视频文件（即对应的pkl文件存在）
    filelist = glob.glob(f"{output_dir}/*.wav")
    filelist = [video for video in filelist if not os.path.exists(os.path.join(feature_path, os.path.splitext(os.path.basename(video))[0] + '.pkl'))]
    print(f"Total audio to process : {len(filelist)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载预训练的VGGish模型
    vggish_model = VGGISH.get_model()
    vggish_model = vggish_model.to(device)
    vggish_model.eval()
    vggish_input_proc = VGGISH.get_input_processor()
    input_sr = VGGISH.sample_rate

    for wav in tqdm(filelist):
        try:
            waveform, sr = torchaudio.load(wav)
        except Exception as e:
            with open(error_log_feature, "a") as f:
                f.write("Error loading waveform: {}\n".format(e))
                continue
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = torchaudio.functional.resample(waveform, sr, input_sr)
        waveform = waveform.squeeze(0)
        waveform = waveform.to(device)
        with torch.no_grad():
            audio_features = vggish_model(vggish_input_proc(waveform)).cpu().numpy()

        flag = False
        if audio_features.shape[0]!= 10:
            with open(error_log_feature, "a") as f:
                f.write("Error processing waveform: {} audio_features shape is not 10\n".format(wav))
                f.flush()
            flag = True
        if flag:
            continue

        # 保存特征
        save_dict = {"audio_features": audio_features}
        saved_path = os.path.join(feature_path, os.path.splitext(os.path.basename(wav))[0] + '.pkl')
        with open(saved_path, 'wb') as f:
            pickle.dump(save_dict, f)
        f.close()




        
