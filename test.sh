

python ./main.py \
--gpu 0 \
--clip_gradient 0.5 \
--snapshot_pref "./checkpoint/" \
--logs_dir "./logs/" \
--batch_size 64 \
--test_batch_size 16 \
--print_freq 1 \
--avepm false \
--ave true \
--preprocess 'None' \
--audio_preprocess 'None' \
--is_select true \
--data_root "./dataset/feature/xxx/" \
--meta_root "./dataset/csv/" \
--v_feature_root "./dataset/feature/aug_xxx/preprocess_visual_feature/" \
--a_feature_root "./dataset/feature/aug_xxx/preprocess_audio_feature/" \
--use_cln true \
--use_saliency true \
--evaluate \
--resume "./checkpoint/xxx.pth.tar" \
--category_num 28 \
--guide "audio"