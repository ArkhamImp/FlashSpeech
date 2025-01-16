#!/bin/bash
# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#SBATCH -o train.out
#SBATCH -p batch --gres=gpu:1
#SBATCH -J tts

#source /home/mingyang/miniconda3/bin/activate flash

######## Build Experiment Environment ###########
exp_dir='egs/tts/NaturalSpeech2'
work_dir='/mnt/lsk_nas/mingyang/FlashSpeech'

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export CUDA_VISIBLE_DEVICES=5
 
######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config_s1.json" #s1 or s2
exp_name="flashspeech_fs"

stage=1
stop_stage=1


######## Features Extraction ###########
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python "${work_dir}"/bins/tts/preprocess.py \
        --config=$exp_config \
        --num_workers=4 
fi

######## Train Model ###########
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    nohup python \
        bins/tts/train_new.py \
        --config=$exp_config \
        --exp_name=$exp_name \
        --log_level debug > train_fs.out 2>&1 &
        # --resume \
        # --checkpoint_path /mnt/lsk_nas/mingyang/FlashSpeech/flashspeech_log/last-v10.ckpt 
fi
