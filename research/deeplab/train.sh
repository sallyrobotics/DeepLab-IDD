
# This script is used to run local test on IDD Dataset. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd /home/sall/CV/tfmodels_server/tfmodels/research

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export CUDA_VISIBLE_DEVICES=1

python deeplab/train.py \
   --logtostderr \
   --training_number_of_steps=90000 \
   --train_split="train" \
   --model_variant="xception_71" \
   --dense_prediction_cell_json="/home/sall/CV/tfmodels_server/tfmodels/research/deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" \
   --atrous_rates=6 \
   --atrous_rates=12 \
   --atrous_rates=18 \
   --output_stride=16 \
   --decoder_output_stride=4 \
   --train_crop_size=513 \
   --train_crop_size=513 \
   --train_batch_size=2 \
   --num_clones=1 \
   --base_learning_rate=1.25e-3 \
   --momentum=0.95 \
   --weight_decay=0.00004 \
   --num_threads=4 \
   --num_readers=4 \
   --fine_tune_batch_norm=true \
   --dataset="anue" \
   --tf_initial_checkpoint=/home/sall/CV/tfmodels_server/tfmodels/research/deeplab/datasets/xception71/model.ckpt  \
   --train_logdir=/home/sall/CV/tfmodels_server/tfmodels/research/deeplab/datasets/hptuning3 \
   --dataset_dir=/mnt/hdd/IDD-semantic/tfrecord-semantic
