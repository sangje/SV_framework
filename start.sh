project_name="vox2_unsupervised_with_vicreg_infonce"

encoder_name="conformer_cat" # conformer_cat | ecapa_tdnn_large | resnet34
embedding_dim=192
projector_dim=1024
loss_name="VICReg+InfoNCE"

dataset="vox2"
num_classes=7205
num_blocks=6
train_csv_path="data/vox2.csv"
checkpoint_path="experiment/Vox1_unsupervised_with_vicreg_infonce/conformer_cat_6_192_vox1/epoch=47_cosine_eer=9.97.ckpt"

input_layer=conv2d2
pos_enc_layer_type=rel_pos # no_pos| rel_pos 
save_dir=experiment/${project_name}/${encoder_name}_${num_blocks}_${embedding_dim}_${dataset}
trial_path=data/vox1_test.txt

mkdir -p $save_dir
#cp start.sh $save_dir
#cp main.py $save_dir
#cp -r module $save_dir
#cp -r wenet $save_dir
#cp -r scripts $save_dir
#cp -r loss $save_dir
echo save_dir: $save_dir

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 main.py \
        --batch_size 64 \
        --num_workers 8 \
        --max_epochs 100 \
        --embedding_dim $embedding_dim \
        --projector_dim $projector_dim \
        --save_dir $save_dir \
        --encoder_name $encoder_name \
        --train_csv_path $train_csv_path \
        --learning_rate 0.001 \
        --encoder_name ${encoder_name} \
        --num_classes $num_classes \
        --trial_path $trial_path \
        --loss_name $loss_name \
        --num_blocks $num_blocks \
        --step_size 4 \
        --gamma 0.5 \
        --weight_decay 0.0000001 \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        --checkpoint_path $checkpoint_path \
        --pairs \
        --aug

