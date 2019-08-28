# This script is used to run local test on IDD Dataset. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./vis.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd /workspace/HOME/tfmodels/research

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="xception_71" \
    --dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=1081 \
    --vis_crop_size=1921 \
    --dataset="anue" \
    --colormap_type="anue" \
    --checkpoint_dir=/workspace/HOME/tfmodels/research/deeplab/datasets/honutrainer/model.ckpt-90000 \
    --vis_logdir=/workspace/HOME/tfmodels/research/deeplab/datasets/honuvis \
    --dataset_dir=/workspace/HOME/anue/tfrecord \
    --max_number_of_iterations=1
