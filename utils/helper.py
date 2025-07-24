import torch
import torch.nn as nn
from typing import List

def get_per_time_loss_cmbs(p: List[torch.Tensor], labels: torch.Tensor, criterion=None):
    if criterion is None:
        # criterion = nn.BCEWithLogitsLoss(reduction='none')  # 确保 loss 形状保持
        criterion = nn.BCELoss(reduction='none')  # 确保 loss 形状保持

    loss_list = []
    for i in range(len(p)):
        loss = criterion(p[i], labels)  # 计算 BCE loss，形状 [b, t, num_classes]
        loss = loss.clone().mean(dim=-1, keepdim=True)  # 取均值，使形状变为 [b, t]
        loss_list.append(loss)  
    return loss_list  # 返回的是 [batch_size] 个 [t, 1] 的张量

def normalize_score_v2(c_list, lambda_param=1):
    """
    对多个贡献分数 c 进行归一化，确保它们相互影响
    :param c_list: 一个包含多个 c (c_a, c_v, c_av) 的 list 或 tuple
    :param lambda_param: 归一化的参数 λ
    :return: 归一化后的贡献分数 (c_a_hat, c_v_hat, c_av_hat)
    """
    # 1. 拼接所有 c 以计算统一的 softmax 归一化因子
    c_all = torch.cat(c_list, dim=-1)  # (batch_size, num_c_a + num_c_v + num_c_av)
    
    # 2. 计算 softmax 分子
    c_exp = torch.exp(lambda_param * c_all)
    
    # 3. 计算共享的归一化因子
    norm_factor = torch.sum(c_exp, dim=-1, keepdim=True)  # (batch_size, 1)
    
    # 4. 分割归一化后的 c
    c_all_hat = c_exp / norm_factor  # 计算归一化得分
    
    # 5. 拆分回原来的 (c_a, c_v, c_av)
    split_sizes = [c.shape[-1] for c in c_list]  # 获取每个 c 的维度
    c_a_hat, c_v_hat, c_av_hat = torch.split(c_all_hat, split_sizes, dim=-1)
    
    return c_a_hat, c_v_hat, c_av_hat

def mutiply_loss_fn(l_a, l_v, l_av, s_a, s_v, s_av):
    """
        当 l_a 小，而 s_a 大，乘积小，损失变小（符合目标）。
        当 l_a 大，而 s_a 小，乘积小，损失变小（符合目标）。
        但如果 l_a 和 s_a 同时小，或者同时大，则乘积变大，损失变大（惩罚）
    """
    return (l_a * s_a + l_v * s_v + l_av * s_av).mean()  # 目标是最小化它

def normalize_score_concat(c_list, lambda_param=1):
    """
    对多个贡献分数 c 进行归一化，确保它们相互影响
    :param c_list: 一个包含多个 c (c_a, c_v, c_av) 的 list 或 tuple
    :param lambda_param: 归一化的参数 λ
    :return: 归一化后的贡献分数 (c_a_hat, c_v_hat, c_av_hat)
    """
    # 1. 拼接所有 c 以计算统一的 softmax 归一化因子
    c_all = torch.cat(c_list, dim=-1)  # (batch_size, num_c_a + num_c_v + num_c_av)
    
    # 2. 计算 softmax 分子
    c_exp = torch.exp(lambda_param * c_all)
    
    # 3. 计算共享的归一化因子
    norm_factor = torch.sum(c_exp, dim=-1, keepdim=True)  # (batch_size, 1)
    
    # 4. 分割归一化后的 c
    c_all_hat = c_exp / norm_factor  # 计算归一化得分
    
    # 5. 拆分回原来的 (c_a, c_v, c_av)
    split_sizes = [c.shape[-1] for c in c_list]  # 获取每个 c 的维度
    c_hat_list = torch.split(c_all_hat, split_sizes, dim=-1)
    
    return c_hat_list

import torch
import torch.nn.functional as F

def MMLoss_fn(p_list, c_list, label):
    """
    p_list: [p_a, p_v, p_av]，每个是 logits，shape: (B, T, C)
    c_list: [c_a, c_v, c_av]，每个是 confidence，shape: (B, T) or (B, T, 1)
    label: LongTensor of shape (B, T, C)，每个时间步的分类标签

    返回：
        total_loss: 总损失
        cls_loss: 所有模态的分类损失和
        conf_loss: 所有模态的置信度损失和
    """
    label = label.long()  # 确保是 long 类型
    # label, _ = label.max(dim=1)  # 取出每个时间步的最大分类标签
    cls_loss = 0.0
    conf_loss = 0.0

    for i in range(3):  # 模态数量固定为 3
        p = p_list[i]
        c = c_list[i]

        # B, T, C = p.shape
        # softmax -> pred prob
        prob = F.softmax(p, dim=-1)  # (B, T, C)

        # 分类损失：-∑ y * log(p)
        log_prob = torch.log(prob + 1e-8)  # 避免 log(0)
        ce_loss = -torch.sum(label * log_prob, dim=-1)  # (B, T)
        cls_loss += ce_loss.mean()

        # 置信度监督：conf ≈ TCP = y·p
        tcp_true = torch.sum(label * prob, dim=-1)  # (B, T)
        tcp_hat = c.squeeze(-1) if c.dim() == 3 else c  # (B, T)
        # tcp_hat = c.squeeze(-1) if c.dim() == 2 else c  # (B, T)
        conf_loss += F.mse_loss(tcp_hat, tcp_true, reduction='mean')

    total_loss = cls_loss + conf_loss
    return total_loss, cls_loss, conf_loss

import torch.nn.functional as F

def MMLoss_fn_v2(p_list, c_list, label_cls, is_event_list, label_bce, lambda_event=1.0):
    """
    p_list: [p_a, p_v, p_av]，每个是分类 logits，shape: (B, T, C)
    c_list: [c_a, c_v, c_av]，每个是置信度，shape: (B, T) or (B, T, 1)
    label_cls: (B, T, C)，分类标签 one-hot
    is_event_list: [e_a, e_v, e_av]，每个是事件打分 logits，shape: (B, T)
    label_bce: (B, T)，是否为事件（二分类标签）
    lambda_event: 权重系数（建议 0.5～1.0）
    
    返回：
        total_loss: 总损失
        cls_loss: 所有模态的分类损失和
        conf_loss: 所有模态的置信度监督损失和
        event_loss: 加权事件性融合后的 BCE 损失
    """
    label_cls = label_cls.long()
    cls_loss = 0.0
    conf_loss = 0.0

    for i in range(3):  # 对 p_a, p_v, p_av
        p = p_list[i]
        c = c_list[i]
        e = is_event_list[i]

        # softmax -> pred prob
        prob = F.softmax(p, dim=-1)  # (B, T, C)

        # 分类损失：-∑ y * log(p)
        log_prob = torch.log(prob + 1e-8)
        ce = -torch.sum(label_cls * log_prob, dim=-1)  # (B, T)
        cls_loss += ce.mean()

        # TCP 对齐监督（conf ≈ y·p）
        tcp_true = torch.sum(label_cls * prob, dim=-1)  # (B, T)
        tcp_hat = c.squeeze(-1) if c.dim() == 3 else c  # (B, T)
        conf_loss += F.mse_loss(tcp_hat, tcp_true, reduction='mean')

    # ➕ 事件性监督项（e_p 与 label_bce）
    with torch.no_grad():
        c_a = c_list[0].detach()
        c_v = c_list[1].detach()
        c_av = c_list[2].detach()
        e_a = is_event_list[0]
        e_v = is_event_list[1]
        e_av = is_event_list[2]
        # e_p: 加权融合事件分数
        e_p = c_a * e_a + c_v * e_v + c_av * e_av  # (B, T)
        e_p = e_p.squeeze(-1)  # (B, T)

    event_loss = F.binary_cross_entropy_with_logits(
        e_p, label_bce.float(), reduction='mean'
    )

    total_loss = cls_loss + conf_loss + lambda_event * event_loss
    return total_loss, cls_loss, conf_loss, event_loss


import matplotlib.pyplot as plt
import os
import numpy as np
import torch
def visualize_saliency_heatmap(sal_map, save_path=None, title='Saliency Heatmap'):
    """
    仅可视化sal_map的热图。
    sal_map: [1, H, W] or [H, W]，Tensor or numpy
    save_path: 保存路径（可选）
    """
    if isinstance(sal_map, torch.Tensor):
        sal_map = sal_map.squeeze().detach().cpu().numpy()  # [H, W]

    plt.figure(figsize=(4, 4))
    plt.imshow(sal_map, cmap='hot')  # 或者 cmap='jet'
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

def total_variation(x):  # x: [B, T, 1, H, W]
    # tv_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
    # tv_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w

from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_frame(frame, resize_size=256, crop_size=224):
    image = Image.fromarray(frame)
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size)
    ])
    return np.array(transform(image))  # 返回 crop 后的 numpy HWC 图像

def preprocess_frame_v1():
    pass

def get_video_frame(video_path, frame_index):
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"读取第 {frame_index} 帧失败: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def visualize_saliency_on_cropped_frame(sal_map, cropped_frame, save_path):
    saliency_np = sal_map.squeeze().cpu().numpy()  # [7, 7]
    H, W, _ = cropped_frame.shape
    saliency_resized = cv2.resize(saliency_np, (W, H), interpolation=cv2.INTER_LINEAR)
    saliency_resized = (saliency_resized - saliency_resized.min()) / (saliency_resized.ptp() + 1e-6)
    saliency_uint8 = np.uint8(255 * saliency_resized)
    saliency_color = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(cropped_frame, 0.6, saliency_color, 0.4, 0)
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Saliency Overlay")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

import sys
sys.path.append('/data1/cy/MTN/')
# from scripts.encode import resize_and_pad
import torchvision.transforms.functional as TF

def resize_and_pad(image, target_size=224):
    w, h = image.size
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

def visualize_saliency_with_resize_pad(sal_map, raw_frame, save_path, target_size=224):
    """
    raw_frame: numpy array (H, W, 3), 原始图像 (RGB)
    sal_map: torch.Tensor [1, 7, 7]，显著图
    save_path: 要保存的路径
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    # === Step 1: 复现 resize_and_pad ===
    img = Image.fromarray(raw_frame)
    processed = resize_and_pad(img, target_size)  # 得到 PIL 图像
    processed_np = np.array(processed)  # [H, W, 3]

    # === Step 2: 处理 saliency map ===
    sal_map_np = sal_map.squeeze().cpu().numpy()  # [7, 7]
    saliency_resized = cv2.resize(sal_map_np, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    saliency_resized = (saliency_resized - saliency_resized.min()) / (saliency_resized.ptp() + 1e-6)
    saliency_uint8 = np.uint8(255 * saliency_resized)
    saliency_color = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB)

    # === Step 3: Overlay ===
    overlay = cv2.addWeighted(processed_np, 0.6, saliency_color, 0.4, 0)
    
    # === Step 4: Save ===
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Saliency Overlay")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
