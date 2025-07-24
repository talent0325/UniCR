import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureSaliency(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sal = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, T, H, W, C=512]
        B, T, H, W, C = x.shape
         # 调整为 [B*T, C, H, W] 以便输入到Conv2d中
        x_reshaped = x.permute(0, 1, 4, 2, 3).contiguous().view(B * T, C, H, W)
        sal_map = self.sal(x_reshaped)  # 输出: [B*T, 1, 7, 7]

        # 加权：广播乘法
        # x_weighted = x_reshaped * sal_map  # [B*T, C, H, W]
        x_weighted = x_reshaped * sal_map + x_reshaped  # residually connected

        # reshape 回原始形状
        x_out = x_weighted.view(B, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous()  # [B, T, 7, 7, 512]
        sal_map = sal_map.view(B, T, 1, H, W)  # [B, T, 1, 7, 7]

        return x_out, sal_map

class PerFrameSaliency(nn.Module):
    """
    其实是每个second的特征图的saliency map
    """
    def __init__(self, in_channels):
        super().__init__()
        self.sal = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            # nn.Sigmoid()
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.sal.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
    def forward(self, x):  # x: [B, T, 7, 7, 512]
        B, T, H, W, C = x.shape
        x_att_list = []
        sal_map_list = []

        for t in range(T):
            frame = x[:, t]  # [B, 7, 7, 512]
            frame = frame.permute(0, 3, 1, 2).contiguous()  # [B, 512, 7, 7]

            sal_map = self.sal(frame)  # [B, 1, 7, 7]
            # frame_att = frame * sal_map  # 加权特征 [B, 512, 7, 7]

            # B, _, H, W = sal_map.shape
            # sal_map = sal_map.view(B, -1)
            # sal_map = F.softmax(sal_map / 6, dim=-1)
            # sal_map = sal_map.view(B, 1, H, W)

            frame_att = frame * sal_map + frame # residually connected [B, 512, 7, 7]


            # 存储
            x_att_list.append(frame_att.permute(0, 2, 3, 1))  # [B, 7, 7, 512]
            sal_map_list.append(sal_map)  # [B, 1, 7, 7]

        # 拼接时间维度
        x_att = torch.stack(x_att_list, dim=1)  # [B, T, 7, 7, 512]
        sal_maps = torch.stack(sal_map_list, dim=1)  # [B, T, 1, 7, 7]

        return x_att, sal_maps

import torch
import torch.nn as nn
import torch.nn.functional as F
# from nets.net_trans import MMIL_Net
    
def normalize_score(c, lambda_param=1, k=3):
    """
    对贡献分数进行归一化
    """
    c_exp = torch.exp(lambda_param * c)
    norm_factor = torch.sum(c_exp, dim=-1, keepdim=True)  # 归一化因子
    c_hat = c_exp / norm_factor  # softmax 归一化
    return c_hat

class CLN(nn.Module):
    """
    x: 单模态输入特征
    d: 输入特征的维度
    dh: 隐藏层的维度
    返回的是贡献分数c
    """
    def __init__(self, d=1536, dh=512, is_multi = False):
        super(CLN, self).__init__()
        if is_multi:
            d = 2 * d
        # 定义模型中的层
        self.fc1 = nn.Linear(d, dh)   # 输入层 -> 隐藏层
        self.fc2 = nn.Linear(dh, dh)  # 隐藏层 -> 隐藏层
        self.fc3 = nn.Linear(dh, dh)  # 隐藏层 -> 隐藏层
        self.fc4 = nn.Linear(dh, 1)   # 隐藏层 -> 输出层

        # 定义激活函数
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 归一化

        self.initialize_weights()


    def initialize_weights(self):
        # 遍历模型中的所有层
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # 对卷积层使用He初始化
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # 对全连接层使用Xavier初始化
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层：输入 -> 隐藏层 -> relu
        x = self.sigmoid(self.fc4(x))  # 输出层：隐藏层 -> 输出层 -> sigmoid
        return x


class MBL(nn.Module):
    """
    Multimodal Base Learner
    拼接单模态特征，经过全连接层来融合
    没有用，是因为LAVISH的输出就有融合特征
    """
    def __init__(self, d, num_modality = 2):
        super(MBL, self).__init__()
        
        # 定义模型中的层
        self.fc1 = nn.Linear(d * num_modality, d)   # 输入层 -> 隐藏层
        self.fc2 = nn.Linear(d, 1)      # 隐藏层 -> 输出层

        # 定义激活函数
        self.tanh = nn.Tanh()

    def forward(self, features):

        x = torch.cat(features, dim = 1)
        # 向前传播
        x = self.tanh(self.fc1(x))  # 第一层：输入 -> 隐藏层 -> tanh
        x = self.fc2(x)             # 第二层：隐藏层 -> 输出层
        return x
    
class Predictor(nn.Module):
    """
    定义预测器
    输入是特征，输出是贡献分数
    """
    def __init__(self, d_in = 1536, num_class = 10, is_multi = False):
        super(Predictor, self).__init__()

        # 定义模型中的层
        if is_multi:
            d_in = 2 * d_in
        self.mlp_class = nn.Linear(d_in, 512) # swinv2-Large
        self.mlp_class_2 = nn.Linear(512, num_class)

    def forward(self, x):
        # 向前传播
        x = self.mlp_class(x)
        x = self.mlp_class_2(x)
        # due to BCEWithLogitsLoss
        p = F.softmax(x, dim = -1)

        return p

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class Saliency_CBAM(nn.Module):
    """
    融合 CBAM 的每秒显著性模块：保持输入输出一致
    输入: [B, T, 7, 7, C]
    输出: 加权特征 [B, T, 7, 7, C], 显著性图 [B, T, 1, 7, 7]
    """
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        B, T, H, W, C = x.shape
        x_att_list = []
        sal_map_list = []

        for t in range(T):
            frame = x[:, t]  # [B, 7, 7, C]
            frame = frame.permute(0, 3, 1, 2).contiguous()  # [B, C, 7, 7]

            # CBAM attention
            ca = self.channel_att(frame)        # [B, C, 1, 1]
            frame_ca = frame * ca

            sa = self.spatial_att(frame_ca)     # [B, 1, 7, 7]
            frame_sa = frame_ca * sa

            # Residual connection
            frame_out = frame + frame_sa

            # 存储
            x_att_list.append(frame_out.permute(0, 2, 3, 1))  # [B, 7, 7, C]
            sal_map_list.append(sa)  # [B, 1, 7, 7]

        # 时间拼接
        x_att = torch.stack(x_att_list, dim=1)       # [B, T, 7, 7, C]
        sal_maps = torch.stack(sal_map_list, dim=1)  # [B, T, 1, 7, 7]

        return x_att, sal_maps


class Saliency_SelfAttention(nn.Module):
    """
    利用空间自注意力建模的显著性模块。
    输入: [B, T, 7, 7, C]
    输出: [B, T, 7, 7, C] (加权后特征), [B, T, 1, 7, 7] (显著性图)
    """
    def __init__(self, in_channels, heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
        self.proj = nn.Linear(in_channels, 1)  # 显著性图预测用
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, T, 7, 7, C]
        B, T, H, W, C = x.shape
        x_att_list = []
        sal_map_list = []

        for t in range(T):
            frame = x[:, t]  # [B, 7, 7, C]
            frame_flat = frame.view(B, H * W, C)  # [B, 49, C]

            # Self-attention
            attn_out, _ = self.attn(frame_flat, frame_flat, frame_flat)  # [B, 49, C]
            attn_out = attn_out.view(B, H, W, C).contiguous()  # [B, 7, 7, C]
            
            # Residual connection
            frame_out = frame + attn_out  # [B, 7, 7, C]

            # Saliency map prediction
            sal_map = self.proj(attn_out.view(B, -1, C))  # [B, 49, 1]
            sal_map = sal_map.view(B, 1, H, W)
            sal_map = self.sigmoid(sal_map)  # [B, 1, 7, 7]

            # Apply saliency to attention output
            weighted = frame_out.permute(0, 3, 1, 2) * sal_map + frame_out.permute(0, 3, 1, 2)
            x_att_list.append(weighted.permute(0, 2, 3, 1))  # [B, 7, 7, C]
            sal_map_list.append(sal_map)  # [B, 1, 7, 7]

        x_att = torch.stack(x_att_list, dim=1)       # [B, T, 7, 7, C]
        sal_maps = torch.stack(sal_map_list, dim=1)  # [B, T, 1, 7, 7]

        return x_att, sal_maps


class PerFrameSaliency_Hybrid(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        self.conv_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)
        self.sal_proj = nn.Linear(in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, T, 7, 7, C]
        B, T, H, W, C = x.shape
        x_att_list, sal_list = [], []

        for t in range(T):
            frame = x[:, t].permute(0, 3, 1, 2)  # [B, C, 7, 7]
            conv_feat = self.conv_refine(frame)  # [B, C, 7, 7]

            flat = conv_feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # [B, 49, C]
            attn_out, _ = self.attn(flat, flat, flat)  # [B, 49, C]
            out = self.norm(flat + attn_out).view(B, H, W, C)

            # saliency
            sal_map = self.sal_proj(attn_out).view(B, 1, H, W)
            sal_map = self.sigmoid(sal_map)

            # final output
            weighted = conv_feat * sal_map + conv_feat
            x_att_list.append(weighted.permute(0, 2, 3, 1))  # [B, 7, 7, C]
            sal_list.append(sal_map)

        x_att = torch.stack(x_att_list, dim=1)       # [B, T, 7, 7, C]
        sal_maps = torch.stack(sal_list, dim=1)      # [B, T, 1, 7, 7]

        return x_att, sal_maps

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height=7, width=7):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4 for 2D positional encoding.")
        
        self.height = height
        self.width = width
        c = channels // 2

        pe = torch.zeros(channels, height, width)

        # 计算 div_term
        div_term = torch.exp(torch.arange(0., c, 2) * -(math.log(10000.0) / c))

        # 水平位置编码（宽度方向）
        pos_w = torch.arange(0., width).unsqueeze(1)  # [W, 1]
        pe[0:c:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).expand(-1, height, -1)
        pe[1:c:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).expand(-1, height, -1)

        # 垂直位置编码（高度方向）
        pos_h = torch.arange(0., height).unsqueeze(1)  # [H, 1]
        pe[c::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).expand(-1, -1, width)
        pe[c+1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).expand(-1, -1, width)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, C, H, W]

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class PerFrameSaliency_AttnPE(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=heads, batch_first=True)
        self.pos_encoding = PositionalEncoding2D(in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.sal_proj = nn.Linear(in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, T, 7, 7, C]
        B, T, H, W, C = x.shape
        x_att_list, sal_list = [], []

        for t in range(T):
            frame = x[:, t]  # [B, 7, 7, C]
            frame = frame.permute(0, 3, 1, 2)  # [B, C, H, W]
            frame = self.pos_encoding(frame)  # + positional encoding
            frame = frame.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # [B, 49, C]

            attn_out, _ = self.attn(frame, frame, frame)  # [B, 49, C]
            out = self.norm(frame + attn_out)  # residual + norm
            out_2d = out.view(B, H, W, C)

            # saliency map
            sal_map = self.sal_proj(out).view(B, 1, H, W)
            sal_map = self.sigmoid(sal_map)

            # Apply saliency
            weighted = out_2d.permute(0, 3, 1, 2) * sal_map + out_2d.permute(0, 3, 1, 2)
            x_att_list.append(weighted.permute(0, 2, 3, 1))
            sal_list.append(sal_map)

        x_att = torch.stack(x_att_list, dim=1)       # [B, T, 7, 7, C]
        sal_maps = torch.stack(sal_list, dim=1)      # [B, T, 1, 7, 7]

        return x_att, sal_maps
