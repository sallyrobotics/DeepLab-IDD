# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="/home/f2015078"

# Root path for ANUE dataset.
ANUE_ROOT="/anue"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${ANUE_ROOT}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/build_anue_data.py"

echo "Converting ANUE dataset..."
python "${BUILD_SCRIPT}" \
  --anue_root="${ANUE_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
