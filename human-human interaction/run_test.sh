python \
inference.py \
--gpu_list 0 \
--dataset_dir /data/PEV \
--dataset_list_file data/PEV/testlist1.txt \
--checkpoint_dir ./checkpoints/ \
--checkpoint_file ./checkpoints/model.ckpt-3001 \
--vis_flag True \
--vis_freq 300 \
--visual_dir ./inference_vis/ \
--cls_loss_weight 1e0 \
--parsing_loss_weight 1e0 \
--pixel_loss_weight 1e0 \
--smooth_loss_weight 1e-3 \
--base_net_weight_reg 1e-3 \
--attention_net_weight_reg 1e-2 \
--motion_net_weight_reg 1e-3 \
--interaction_net_weight_reg 1e-1 \
--to_obs_loss pixel_loss \
--num_class 8