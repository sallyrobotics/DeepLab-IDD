import glob
import math
import os.path
import re
import sys
import build_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('anue_root',
                           '~/anue',
                           'ANUE dataset root folder.')

tf.app.flags.DEFINE_string(
    'output_dir',
    '~/anue/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


_NUM_SHARDS = 10

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labellevel3Ids',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])


def _get_files(data, dataset_split):
  """Gets files for the specified data type and dataset split.
  Args:
    data: String, desired data ('image' or 'label').
    dataset_split: String, dataset split ('train', 'val', 'test')
  Returns:
    A list of sorted file names or None when getting label for
      test set.
  """
  if data == 'label' and dataset_split == 'test':
    return None
  pattern = '*{}.{}'.format(_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
  search_files = os.path.join(
      FLAGS.anue_root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
  filenames = glob.glob(search_files)
  return sorted(filenames)


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = _get_files('image', dataset_split)
  label_files = _get_files('label', dataset_split)

  num_images = len(image_files)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    shard_filename = '{0}-{1:05d}-of-{2:05d}.tfrecord'.format(dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.output_dir, shard_filename)

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)

      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(i + 1, num_images, shard_id))
        sys.stdout.flush()

        # Read the image.
        image_data = tf.gfile.FastGFile(image_files[i], 'rb').read()
        height, width = image_reader.read_image_dims(image_data)

        # Read the semantic segmentation annotation.
        seg_data = tf.gfile.FastGFile(label_files[i], 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')

        # Convert to tf example.
        re_match = _IMAGE_FILENAME_RE.search(image_files[i])
        if re_match is None:
          raise RuntimeError('Invalid image filename: ' + image_files[i])
        filename = os.path.basename(re_match.group(1))
        example = build_data.image_seg_to_tfexample(
            image_data, filename, height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train' and 'val' sets for now.
  for dataset_split in ['train', 'val']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
