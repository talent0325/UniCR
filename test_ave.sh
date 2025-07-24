# cd /data1/cy/MTN/cmbs
# TIME=$(date "+%Y-%m-%d-%H-%M-%S")
# STR="use multiply_rank_loss_fn and 3 tanh in cln"

# echo "========================= $STR ===========================" | tee -a /data1/cy/MTN/logs/$TIME.log

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
--batch_size 64 \
--test_batch_size 16 \
--print_freq $FREQ \
--avepm false \
--ave true \
--warm_up $WARMUP \
--preprocess 'None' \
--audio_preprocess 'None' \
--is_select $FLAG \
--data_root "./dataset/feature/ave/" \
--meta_root "./dataset/csv/" \
--v_feature_root "./dataset/feature/aug_ave/preprocess_visual_feature/" \
--a_feature_root "./dataset/feature/aug_ave/preprocess_audio_feature/" \
--use_cln true \
--use_saliency true \
--evaluate \
--resume "./checkpoint/2025-07-17-12-43-42/epoch_31_80.072.pth.tar" \
--category_num 28 \
--guide "audio"