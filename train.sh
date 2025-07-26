# !/bin/bash

# select datasets and run experiments
# if is_select is true, you choose the S-xxx dataset to train and test
# if ave is true, you use the AVE dataset to train and test
# if avepm is true, you use the AVE-PM dataset to train and test
# Note: you should set the correct data_root, meta_root, v_feature_root, a_feature_root, category_num, and preprocess according to your own dataset

python ./main.py \
--gpu 1 \
--lr 0.0001 \
--lr_cln 0.0001 \
--clip_gradient 0.5 \
--snapshot_pref "./checkpoint/" \
--logs_dir "./logs/" \
--n_epoch 200 \
--b 64 \
--test_batch_size 16 \
--print_freq 100 \
--ave true \
--avepm false \
--warm_up 2 \
--preprocess "inception" \
--audio_preprocess "None" \
--is_select true \
--data_root "./dataset/feature/ave/features/" \
--meta_root "./dataset/csv/" \
--v_feature_root "./dataset/feature/aug_ave/preprocess_visual_feature/" \
--a_feature_root "./dataset/feature/aug_ave/preprocess_audio_feature/" \
--use_cln true \
--use_saliency true \
--category_num 10

