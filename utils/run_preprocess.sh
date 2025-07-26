#!/bin/bash
set -e  # 一旦出错就停止运行


# -----------------------------
# Visual Preprocessing
# # -----------------------------
SEED=42
python ./utils/visual_preprocess.py \
    --csv_path "dataset\csv\ave\ave_annotations_with_labels.csv" \
    --frame_dir "dataset\frames\ave" \
    --video_root "dataset\videos\ave" \
    --feature_root "dataset\features\ave\preprocess_visual_feature" \
    --feature_root "./dataset/feature/aug_ave/preprocess_visual_feature/" \
    --random_seed $SEED \
    --token "sample_id"
