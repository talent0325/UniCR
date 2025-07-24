for v in "None" 
do
    for a in "None"
    do
        python ./main.py \
        --gpu 0 \
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
        --warm_up 10 \
        --preprocess $v \
        --audio_preprocess $a \
        --is_select false \
        --data_root "./dataset/feature/ave/" \
        --meta_root "./dataset/csv/" \
        --v_feature_root "./dataset/feature/aug_ave/preprocess_visual_feature/" \
        --a_feature_root "./dataset/feature/aug_ave/preprocess_audio_feature/" \
        --use_cln true \
        --use_saliency true \
        --category_num 28 \
        --guide "audio"
    done
done
