# sh ./scripts/train.sh

EXP_NAME=demo_exp
CONFIG_PATH=./config/FI/FI_res50.yaml
RESUME_PATH=''
SEED=1234

CUDA_VISIBLE_DEVICES=0 \
python ./scripts/train.py \
    --exp_name $EXP_NAME \
    --config_path $CONFIG_PATH \
    --log_path './logs/training/' \
    --seed $SEED

# --resume_path $RESUME_PATH \