

# FLAG=true
FLAG=false
WARMUP=10
# WARMUP=50
# EPOCHS=50
EPOCHS=200

LR=0.0001
LR_CLN=0.0001

FREQ=100

python ./main.py \
--gpu 0 \
--lr $LR \
--lr_cln $LR_CLN \
--clip_gradient 0.5 \
--snapshot_pref "./checkpoint/" \
--logs_dir "./logs/" \
--n_epoch $EPOCHS \
--b 64 \
--test_batch_size 16 \
--print_freq $FREQ \
--avepm false \
--ave true \
--warm_up $WARMUP \
--preprocess 'random_crop' \
--audio_preprocess 'None' \
--is_select $FLAG \
--data_root "./dataset/feature/ave/" \
--meta_root "./dataset/csv/" \
--v_feature_root "./dataset/feature/aug_ave/preprocess_visual_feature/" \
--a_feature_root "./dataset/feature/aug_ave/preprocess_audio_feature/" \
--use_cln true \
--use_saliency true \
--category_num 28 \
