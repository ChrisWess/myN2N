import tensorflow as tf
from typing import Callable


def parse_tfrecord_tf(record):
    """Define the feature dict containing the feature keys that were used in dataset_tool_tf to
    create the tfrecord file. Each example contains the data of 1) the image shape and 2) the pixel data."""
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    # Decode the binary data to the byte (uint8) pixel color values.
    data = tf.decode_raw(features['data'], tf.uint8)
    # Reshape the pixels to fit the shape of this example image.
    return tf.reshape(data, features['shape'])


# HWC = height, width, channel
# [c,h,w] -> [h,w,c]
# [0,1,2] -> [1,2,0]
def chw_to_hwc(x):
    return tf.transpose(x, perm=[1, 2, 0])


# [h,w,c] -> [c,h,w]
def hwc_to_chw(x):
    return tf.transpose(x, perm=[2, 0, 1])


def resize_small_image(x):
    """If the image is smaller than 256*256, extend it to fit to the size of 256*256."""
    shape = tf.shape(x)
    return tf.cond(
        tf.logical_or(
            tf.less(shape[2], 256),
            tf.less(shape[1], 256)
        ),
        # Resize if too small.
        true_fn=lambda: hwc_to_chw(tf.image.resize_images(chw_to_hwc(x), size=[256, 256], method=tf.image.ResizeMethod.BICUBIC)),
        # Otherwise just make sure image is of type float32.
        false_fn=lambda: tf.cast(x, tf.float32)
     )


def random_crop_noised_clean(x: tf.Tensor, add_noise: Callable) -> tuple:
    """Crop image and return a 3-tuple with this crop, 2x with added random noise and once clean.
    result means: noisy input, noisy target, clean target."""
    # Randomly crops an 256*256 image to fit into the input layer and normalizes RGB pixel values.
    cropped = tf.random_crop(resize_small_image(x), size=[3, 256, 256]) / 255.0 - 0.5
    # TODO: Make more crops as training data.
    return add_noise(cropped), add_noise(cropped), cropped


def create_dataset(train_tfrecords: str, minibatch_size: int, add_noise: Callable) -> iter:
    print('Setting up dataset source from', train_tfrecords)
    buffer_mb = 256
    num_threads = 2
    dset = tf.data.TFRecordDataset(train_tfrecords, compression_type='', buffer_size=buffer_mb << 20)
    dset = dset.repeat()
    buf_size = 1000
    # Prefetches a set of data into a buffer with the given size (to not fetch after every iteration).
    dset = dset.prefetch(buf_size)
    # Shuffle data for each training epoch.
    dset = dset.shuffle(buffer_size=buf_size)
    # Parses the binary tfrecords data to the actual image tensors with uint8 pixel color values.
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
    # Random crop the all images and add noise.
    dset = dset.map(lambda x: random_crop_noised_clean(x, add_noise))
    dset = dset.batch(minibatch_size)
    it = dset.make_one_shot_iterator()
    return it

