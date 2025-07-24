#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle
from torchaudio.prototype.pipelines import VGGISH
from tqdm import tqdm
import logging

# 配置logger
def setup_logger():
    logger = logging.getLogger('EventTemplateGenerator')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('log.out')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

class EventTemplateGenerator:
    def __init__(self, annotation_csv, video_root_dir, output_dir="templates",
                 n_clusters=3):
        """
        初始化事件模板生成器

        参数:
            annotation_csv: 注释文件路径
            video_root_dir: 视频文件根目录
            output_dir: 输出模板保存目录
            n_clusters: 每个类别的聚类数量
        """
        # 读取CSV并去除重复的video_id行
        self.df = pd.read_csv(annotation_csv)
        self.df = self.df.drop_duplicates(subset=['video_id'], keep='first')
        
        self.video_dir = video_root_dir
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取唯一的类别列表
        self.categories = self.df['category'].unique()
        # music_classes = [4, 5, 10, 81, 84, 86, 92, 93, 102, 106, 108, 118, 120, 196, 346, 347, 348, 349]
        music_classes = [0, 1, 3, 12, 18, 26, 27]

        self.categories = [category for category in self.categories if category not in music_classes]
        
        # 初始化VGGish模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        self.vggish_model = VGGISH.get_model()
        self.vggish_model = self.vggish_model.to(self.device)
        self.vggish_model.eval()
        self.vggish_input_proc = VGGISH.get_input_processor()
        self.input_sr = VGGISH.sample_rate
    
    def extract_audio_features(self, waveform, sr):
        """
        使用VGGish网络提取音频特征
        
        参数:
            waveform: 音频波形 (torch.Tensor)
            sr: 原始采样率
            
        返回:
            特征向量 (numpy.ndarray)
        """
        # 确保输入是单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到VGGish所需采样率
        waveform = torchaudio.functional.resample(waveform, sr, self.input_sr)
        waveform = waveform.squeeze(0)
        
        # 移动到指定设备
        waveform = waveform.to(self.device)
        
        # 使用VGGish提取特征
        with torch.no_grad():
            features = self.vggish_model(self.vggish_input_proc(waveform))
        
        # 如果特征有多个时间步，取均值作为整体特征
        if features.shape[0] > 1:
            features = torch.mean(features, dim=0, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def cluster_category_samples(self, category):
        """
        对指定类别的样本进行聚类
        
        参数:
            category: 事件类别编号
            
        返回:
            聚类结果字典，包含每个簇的代表性样本
        """
        logger.info(f"处理类别: {category}")
        
        # 筛选符合条件的样本：相同类别、无BGM、音频相关
        # filtered_df = self.df[(self.df['category'] == category) & 
        #                    (self.df['haveBGM'] == 0) & 
        #                    (self.df['audio_irrelevant'] == 0)]
        filtered_df = self.df[(self.df['category'] == category)]
        if filtered_df.empty:
            logger.warning(f"类别 {category} 没有无BGM样本")
            return None
        
        if len(filtered_df) < self.n_clusters:
            logger.warning(f"类别 {category} 样本数量 ({len(filtered_df)}) 少于聚类数量 ({self.n_clusters})")
            n_clusters = len(filtered_df)
        else:
            n_clusters = self.n_clusters
        
        # 为每个样本提取特征
        features_list = []
        sample_ids = []
        
        for _, row in tqdm(filtered_df.iterrows(), desc=f"提取类别 {category} 特征"):
            try:
                # 获取视频路径
                video_path = os.path.join(self.video_dir, f"{row['video_id']}.mp4")
                
                # 加载音频
                waveform, sample_rate = torchaudio.load(video_path)
                
                # 提取事件时间段
                start_frame = int(row['event_start'] * sample_rate)
                end_frame = int(row['event_end'] * sample_rate)
                
                # 确保起止帧有效
                if start_frame >= end_frame or end_frame > waveform.shape[1]:
                    continue
                
                # 提取事件音频
                event_audio = waveform[:, start_frame:end_frame]
                
                # 提取特征
                features = self.extract_audio_features(event_audio, sample_rate)
                features_list.append(features)
                sample_ids.append(row.name)
            except Exception as e:
                logger.error(f"处理样本 {row['video_id']} 时出错: {e}")
                continue
        
        if len(features_list) < n_clusters:
            logger.warning(f"类别 {category} 有效特征数量 ({len(features_list)}) 少于聚类数量, 改为使用单一模板")
            n_clusters = 1
        
        if len(features_list) == 0:
            logger.error(f"类别 {category} 没有有效特征")
            return None
        
        # 标准化特征
        features_array = np.array(features_list)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # 对于每个簇，选择距离中心最近的样本
        cluster_templates = {}
        
        for cluster_id in range(n_clusters):
            # 获取当前簇的样本索引
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # 计算每个样本到簇中心的距离
            cluster_samples = features_scaled[cluster_indices]
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_samples - center, axis=1)
            
            # 获取最近的样本索引
            nearest_idx = cluster_indices[np.argmin(distances)]
            sample_id = sample_ids[nearest_idx]
            
            # 获取对应的样本行
            sample = filtered_df.iloc[filtered_df.index == sample_id].iloc[0]
            
            # 加载样本音频
            video_path = os.path.join(self.video_dir, f"{sample['video_id']}.mp4")
            waveform, sample_rate = torchaudio.load(video_path)
            
            # 提取事件时间段
            start_frame = int(sample['event_start'] * sample_rate)
            end_frame = int(sample['event_end'] * sample_rate)
            event_audio = waveform[:, start_frame:end_frame]
            
            # 保存到结果中
            cluster_templates[cluster_id] = {
                'audio': event_audio,
                'sample_rate': sample_rate,
                'video_id': sample['video_id'],
                'event_start': sample['event_start'],
                'event_end': sample['event_end']
            }
            
            logger.info(f"  簇 {cluster_id}: 选择样本 {sample['video_id']}, 长度: {sample['event_end'] - sample['event_start']:.2f}秒")
        
        # 计算聚类质量
        if n_clusters > 1:
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            logger.info(f"  轮廓系数: {silhouette_avg:.4f}")
        
        return cluster_templates
    
    def process_all_categories(self):
        """
        处理所有类别，为每个类别生成模板
        """
        all_templates = {}
        
        for category in self.categories:
            templates = self.cluster_category_samples(category)
            if templates:
                all_templates[category] = templates
        
        # 保存结果
        output_path = os.path.join(self.output_dir, 'event_templates.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(all_templates, f)
        
        logger.info(f"已保存所有模板到 {output_path}")
        
        # 保存模板元数据
        metadata = {}
        for category, templates in all_templates.items():
            metadata[category] = {}
            for cluster_id, template in templates.items():
                metadata[category][cluster_id] = {
                    'video_id': template['video_id'],
                    'event_start': template['event_start'],
                    'event_end': template['event_end'],
                    'duration': template['event_end'] - template['event_start']
                }
        
        metadata_path = os.path.join(self.output_dir, 'template_metadata.csv')
        # 将嵌套字典转换为DataFrame
        metadata_rows = []
        for category, clusters in metadata.items():
            for cluster_id, info in clusters.items():
                row = {'category': category, 'cluster_id': cluster_id}
                row.update(info)
                metadata_rows.append(row)
                
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
        logger.info(f"已保存模板元数据到 {metadata_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='为每个声音事件类别计算最佳模板')
    parser.add_argument('--annotations', type=str, required=True,
                        help='注释CSV文件路径')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频文件目录')
    parser.add_argument('--output_dir', type=str, default='templates',
                        help='输出模板保存目录')
    parser.add_argument('--n_clusters', type=int, default=3,
                        help='每个类别的聚类数量')
    
    args = parser.parse_args()
    
    generator = EventTemplateGenerator(
        args.annotations,
        args.video_dir,
        args.output_dir,
        args.n_clusters
    )
    
    generator.process_all_categories()

if __name__ == '__main__':
    main() 