import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import h5py
from tqdm import tqdm
import glob
from torchvision import models, transforms
import cv2
import numpy as np
import torch
import pandas as pd
from torchaudio.prototype.pipelines import VGGISH
import torchaudio
import os
import glob
from tqdm import tqdm
import pickle
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import torchvision.transforms.functional as TF

def main():
    parser = argparse.ArgumentParser("Extract features from videos")
    parser.add_argument('--folder_path', type=str, default='dataset/data/videos', help='folder path of videos')
    parser.add_argument('--feature_path', type=str, default='dataset/feature/features', help='folder path of output features')

    # parser.add_argument('--annotations', type=str, default='dataset/csvfiles/annotations.csv', help='annotation file')  # xxx
    # parser.add_argument('--class_mapping', type=str, default='dataset/csvfiles/class_name_mapping.csv', help='class mapping file') # xxx

    parser.add_argument('--ave_labels', type=str, default='dataset/data/AVE/data/labels.h5', help='AVE labels file')
    parser.add_argument('--ave_annotations', type=str, default='dataset/data/AVE/data/Annotations.txt', help='AVE annotations file')
    parser.add_argument('--ave_meta', type=str, default='dataset/data/AVE/data/meta_data.csv', help='AVE meta file')
    parser.add_argument('--ave_category', type=str, default='dataset/data/AVE/data/category.txt', help='AVE category file')
    parser.add_argument('--ave_feature_root', type=str, default='dataset/data/AVE/feature', help='AVE feature root')   # S-LM 保存的root

    args = parser.parse_args()
    # MARK: 提取特征
    # 读取文件列表
    folder_path = args.folder_path
    filelist = glob.glob(os.path.join(folder_path, '*.mp4'))
    # 输出文件
    
    feature_path = args.feature_path
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    output_log = f"{feature_path}/error.txt"
    # video名字都是 int
    # 过滤掉已经提取过特征的视频文件（即对应的pkl文件存在）
    filelist = [video for video in filelist if not os.path.exists(os.path.join(feature_path, os.path.splitext(os.path.basename(video))[0] + '.pkl'))]
    print(f"Total videos: {len(filelist)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 3加载预训练的VGG19模型
    vgg19_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    vgg19_model = torch.nn.Sequential(*list(vgg19_model.children())[0])  # 移除分类层
    vgg19_model = vgg19_model.to(device)
    vgg19_model.eval()

    # 加载预训练的VGGish模型
    vggish_model = VGGISH.get_model()
    vggish_model = vggish_model.to(device)
    vggish_model.eval()
    vggish_input_proc = VGGISH.get_input_processor()
    input_sr = VGGISH.sample_rate

    for video_path in tqdm(filelist):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                
                with open(output_log, "a") as f:
                    f.write("Error processing video: {} cannot be opened\n".format(video_path))
                    f.flush()
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            # 29.8帧
            if round(duration) < 10:
                with open(output_log, "a") as f:
                    f.write("Error processing video: {} duration is less than 10s\n".format(video_path))
                    f.flush()
                continue
        except Exception as e:
            with open(output_log, "a") as f:
                f.write("Error processing video: {} exception: {}\n".format(video_path, e))
                f.flush()
            continue

        video_features = []
        for second in range(round(duration)):
            start_frame = int(second * fps)
            end_frame = int((second + 1) * fps)

            frames_in_second = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(start_frame, end_frame):
                ret, frame = cap.read()
                if ret:
                    frames_in_second.append(frame)
            
            # 均匀采样16帧
            if len(frames_in_second) > 0:
                indices = np.linspace(0, len(frames_in_second) - 1, 16, dtype=int)
                sampled_frames = [frames_in_second[i] for i in indices]

                # 提取特征
                features = []
                for frame in sampled_frames:
                    # OpenCV采用的是BGR顺序
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = vgg19_model(input_tensor)
                        output = output.permute(0, 2, 3, 1)
                    features.append(output.cpu().numpy())

                # 计算均值
                avg_feature = np.mean(features, axis=0)
                avg_feature = avg_feature.squeeze()
            else:
                # 错误处理
                with open(output_log, "a") as f:
                    f.write("Error processing video: {} in second {}\n".format(video_path, second))
                    f.flush()
                avg_feature = np.zeros((7, 7, 512))  # 或其他默认值
                pass
            video_features.append(avg_feature)
        video_features = np.array(video_features)

        # 提取音频特征
        waveform, sr = torchaudio.load(video_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = torchaudio.functional.resample(waveform, sr, input_sr)
        waveform = waveform.squeeze(0)
        waveform = waveform.to(device)
        with torch.no_grad():
            audio_features = vggish_model(vggish_input_proc(waveform)).cpu().numpy()

        # 再检查一次
        flag = False
        if video_features.shape[0]!= 10:
            with open(output_log, "a") as f:
                f.write("Error processing video: {} video_features shape is not 10\n".format(video_path))
                f.flush()
            flag = True

        if audio_features.shape[0] != 10:
            with open(output_log, "a") as f:
                f.write("Error processing video: {} audio_features shape is not 10\n".format(video_path))
                f.flush()
            flag = True
        if flag:
            continue

        # 保存特征到文件
        saved_dict = {
            "video_features": video_features,
            "audio_features": audio_features
        }
        saved_path = os.path.join(feature_path, os.path.basename(video_path).split(".")[0] + ".pkl")
        with open(saved_path, "wb") as f:
            pickle.dump(saved_dict, f)

        f.close()
    # MARK: 得到label
    with h5py.File(args.ave_labels, 'r') as label_f:
        ave_labels = label_f['avadataset']
    
        ave_old_labels = []
        with open(args.ave_annotations, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("&")
                category = line[0]
                onset = line[3]
                offset = line[4]
                ave_old_labels.append((category, onset, offset))
        
        ave_new_labels = []
        for label in ave_labels:
            # 把one hot转换为数字
            tmp = []
            for second in label:
                category = list(second).index(1)
                tmp.append(category)
            
            ave_new_labels.append(tmp)
    # 把新的类别塞进提取的pickle文件里
    ave_meta = pd.read_csv(args.ave_meta)
    categories = ave_meta['category'].unique().tolist()
    categories.sort()
    with open(args.ave_category, 'w') as f:
        for category in categories:
            f.write(category+'\n')
    look_up_table = {}
    for i, category in enumerate(categories):
        look_up_table[category] = i


    # process_row
    for _, row in tqdm(ave_meta.iterrows(), total=len(ave_meta)):
        category_idx = look_up_table[row['category']]
        onset = int(row['onset'])
        offset = int(row['offset'])
        onehot_label = np.zeros((10, len(categories)))
        for i in range(onset, offset):
            onehot_label[i][category_idx] = 1
        video_id = row['video_id']
        file_path = os.path.join(args.feature_path, video_id +'.pkl')
        output_path = os.path.join(args.ave_feature_root, video_id +'.pkl')
        if not os.path.exists(file_path):
            print("missing file: {}".format(video_id))
            continue

        try:
            # 读取全部的，保存到副本，修改label写到dataset/feature/select/ave/下
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            tmp = data
            tmp['onehot_labels'] = onehot_label
            with open(output_path, "wb") as of:
                pickle.dump(tmp, of)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    main()