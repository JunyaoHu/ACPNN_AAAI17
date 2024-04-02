# sh ./scripts/train.sh

EXP_NAME=demo_exp
#CONFIG_PATH=./config/emotion6/emotion6_vgg19_bn.yaml
CONFIG_PATH=./config/emotion6/emotion6_sentibank.yaml
RESUME_PATH=''
SEED=42

CUDA_VISIBLE_DEVICES=0 \
python ./scripts/train.py \
    --exp_name $EXP_NAME \
    --config_path $CONFIG_PATH \
    --log_path './logs/training/' \
    --seed $SEED

# --resume_path $RESUME_PATH \