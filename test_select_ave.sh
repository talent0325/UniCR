python ./main.py \
--gpu 1 \
--lr 0.0001 \
--lr_cln 0.0001 \
--clip_gradient 0.5 \
--snapshot_pref "./checkpoint/" \
--logs_dir "./logs/" \
--n_epoch 200 \
--batch_size 64 \
--test_batch_size 16 \
--print_freq 100 \
--ave true \
--avepm false \
--warm_up 5 \
--preprocess "inception" \
--audio_preprocess "None" \
--is_select true \
--data_root "/data1/yyp/AVEL/generalization_exp_data/ave_selected/features/" \
--meta_root "./dataset/csv/" \
--v_feature_root "./dataset/feature/aug_ave/preprocess_visual_feature/" \
--a_feature_root "./dataset/feature/aug_ave/preprocess_audio_feature/" \
--use_cln false \
--use_saliency true \
--evaluate \
--resume "./checkpoint/2025-07-21-03-49-11/epoch_7_86.284.pth.tar" \
--category_num 10


