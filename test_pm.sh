# cd /data1/cy/MTN/cmbs
# TIME=$(date "+%Y-%m-%d-%H-%M-%S")
# STR="use multiply_rank_loss_fn and 3 tanh in cln"

# echo "========================= $STR ===========================" | tee -a /data1/cy/MTN/logs/$TIME.log

FLAG=true
# FLAG=false
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
--ave false \
--avepm true \
--warm_up $WARMUP \
--preprocess 'inception' \
--audio_preprocess 'None' \
--is_select $FLAG \
--data_root "/data1/yyp/AVEL/generalization_exp_data/pm_selected/features/" \
--meta_root "./dataset/csv/" \
--v_feature_root "./dataset/feature/aug_pm/preprocess_visual_feature/" \
--a_feature_root "./dataset/feature/aug_pm/preprocess_audio_feature/" \
--use_cln true \
--use_saliency true \
--evaluate \
--resume "./checkpoint/2025-06-12-18-03-37/epoch_5_86.014.pth.tar" \
--guide "audio" \
--category_num 10 \
--bgm "None"
