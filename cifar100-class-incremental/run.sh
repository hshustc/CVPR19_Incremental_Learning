mkdir data
mkdir checkpoint
#iCaRL
CUDA_VISIBLE_DEVICES=0 python class_incremental_cifar100.py \
    --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 \
    --resume --rs_ratio 0.0 \
    --random_seed 1993 \
    --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_cifar100 \
    2>&1 | tee log_seed_1993_rs_ratio_0.0_class_incremental_cifar100_nb_cl_fg_50_nb_cl_10_nb_protos_20.txt

#Cosine Normalization (Mimic Scores)
CUDA_VISIBLE_DEVICES=1 python class_incremental_cosine_cifar100.py \
    --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 \
    --resume --rs_ratio 0.0 --imprint_weights --mimic_score \
    --random_seed 1993 \
    --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_MS_IPW_cosine_cifar100 \
    2>&1 | tee log_seed_1993_rs_ratio_0.0_class_incremental_MS_IPW_cosine_cifar100_nb_cl_fg_50_nb_cl_10_nb_protos_20.txt

#Cosine Normalization (Mimic Scores) + Less-forget Constraint (Adaptive Loss Weight)
CUDA_VISIBLE_DEVICES=2 python class_incremental_cosine_cifar100.py \
    --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 \
    --resume --rs_ratio 0.0 --imprint_weights \
    --less_forget --lamda 5 --adapt_lamda \
    --random_seed 1993 \
    --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_LFAD_cosine_cifar100 \
    2>&1 | tee log_seed_1993_rs_ratio_0.0_class_incremental_LFAD_cosine_cifar100_nb_cl_fg_50_nb_cl_10_nb_protos_20.txt

#Cosine Normalization (Mimic Scores) + Less-forget Constraint (Adaptive Loss Weight) + Margin Ranking Loss
CUDA_VISIBLE_DEVICES=3 python class_incremental_cosine_cifar100.py \
    --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 \
    --resume --rs_ratio 0.0 --imprint_weights \
    --less_forget --lamda 5 --adapt_lamda \
    --random_seed 1993 \
    --mr_loss --dist 0.5 --K 2 --lw_mr 1 \
    --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100 \
    2>&1 | tee log_seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100_nb_cl_fg_50_nb_cl_10_nb_protos_20.txt

#Cosine Normalization (Mimic Scores) + Less-forget Constraint (Adaptive Loss Weight) + Margin Ranking Loss + Class Balance Finetune
CUDA_VISIBLE_DEVICES=4 python cbf_class_incremental_cosine_cifar100.py \
    --nb_cl_fg 50 --nb_cl 10 --nb_protos 20 \
    --resume --rs_ratio 0.0 --imprint_weights \
    --less_forget --lamda 5 --adapt_lamda \
    --random_seed 1993 \
    --mr_loss --dist 0.5 --K 2 --lw_mr 1 \
    --cb_finetune \
    --ckp_prefix cbf_seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100 \
    2>&1 | tee log_cbf_seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100_nb_cl_fg_50_nb_cl_10_nb_protos_20.txt
