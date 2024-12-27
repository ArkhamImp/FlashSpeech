# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config_s2.json"
exp_name="ns2_ict_normal"
ref_audio="baker/Wave/001910.wav"
checkpoint_path="/aifs4su/mingyang/FlashSpeech/flashspeech_log/epochepoch=3978-stepstep=217010.ckpt"
output_dir="$work_dir/output-fs"
mode="single"

export CUDA_VISIBLE_DEVICES="7"



######## Train Model ###########
python "${work_dir}"/bins/tts/inference.py \
    --config=$exp_config \
    --text='少顷，果真有一小和尚翻墙，黑暗中踩着老禅师的脊背跳进院子。' \
    --mode=$mode \
    --checkpoint_path=$checkpoint_path \
    --ref_audio=$ref_audio \
    --output_dir=$output_dir \
    --inference_step=20

