#!/bin/bash
set -e  # 一旦出错就停止运行

# -----------------------------
# Step 1: Audio Preprocessing
# -----------------------------
python ./utils/audio_preprocess.py \
    --annotation_file "./dataset/csv/ave_annotations_with_labels.csv" \
    --preprocess_type "LMS" \
    --template_path "./dataset/csv/ave/event_templates.pkl" \
    --output_root "./datasets/AVE_Dataset/processed_audios/" \
    --videos_root_dir "./datasets/AVE_Dataset/AVE/" \
    --n_fft 2048 \
    --hop_length 512 \
    --LMS_filter_length 128 \
    --LMS_step_size 0.01 \
    --LMS_iterations 1000 \
    --threshold1 0.3 \
    --threshold2 0.1 \
    --threshold3 0.7 \
    --feature_root "./dataset/feature/aug_ave/preprocess_audio_feature/"

# -----------------------------
# Step 2: Visual Preprocessing
# # -----------------------------
SEED=42
python ./utils/visual_preprocess.py \
    --csv_path "./datasets/AVE_Dataset/AVE_PM_Dataset/final_annotations_with_label_filtered.csv" \
    --frame_dir "./datasets/AVE_Dataset/AVE_PM_Dataset/frames/" \
    --video_root "./datasets/AVE_Dataset/AVE_PM_Dataset/videos/" \
    --feature_root "./dataset/feature/aug_pm/preprocess_visual_feature/" \
    --random_seed $SEED \
    --token "sample_id"
