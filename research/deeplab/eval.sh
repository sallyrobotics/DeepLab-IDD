# This script is used to run local test on IDD Dataset. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./eval.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd /home/sall/CV/tfmodels_server/tfmodels/research

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export CUDA_VISIBLE_DEVICES=0

python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_71" \
    --dense_prediction_cell_json="/home/sall/CV/tfmodels_server/tfmodels/research/deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=1081 \
    --eval_crop_size=1921 \
    --dataset="anue" \
    --checkpoint_dir=/home/sall/CV/tfmodels_server/tfmodels/research/deeplab/datasets/hptuning3 \
    --eval_logdir=/home/sall/CV/tfmodels_server/tfmodels/research/deeplab/datasets/hpeval3 \
    --dataset_dir=/mnt/hdd/IDD-semantic/tfrecord-semantic
