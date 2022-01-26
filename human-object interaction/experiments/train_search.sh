export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../..

python -m torch.distributed.launch --nproc_per_node=4 main.py \
--video_path /data/EPIC-KITCHENS/frames_rgb_flow \
--annotation_path data/EPIC-KITCHENS \
--result_path experiments/results \
--dataset EPIC_Kitchens --modality rgb --n_classes 125 352 \
--sample_duration 16 --sample_step 40 \
--model_depth 18  --basenet_fixed_layers 0 \
--num_mask 6 --mask_sigma 0.1 \
--loss_weight 1e-1 1e-2 1e-3 --loss_vis_weight 1e2 1e-1 1e-2 \
--learning_rate 0.005 --weight_decay 5e-4 --lr_patience 10 \
--batch_size 10 --n_epochs 40 \
--train_list EPIC_train_s1_model_action_labels.txt --val_list EPIC_val_s1_action_labels.txt \
--search --search_list EPIC_train_s1_arch_action_labels.txt --arch_learning_rate 0.0005 --arch_weight_decay 5e-4 \
--n_threads 2 --log_step 20 \
--resume_path /path/to/ckpt_bb_hg/save.pth \
--n_threads 2 --log_step 20 \
--visualize --vis_path experiments/vis 