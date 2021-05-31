import tensorflow as tf
import random
import yaml
import os

config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)
W, H = config['width'], config['height']


def load_dataset(
    data_dir: str,
    batch_size: int = 32,
    buffer_size: int = 1000,
    n_frames: int = 12,
    cache: bool = False,
    train: bool = False,
):
    """
    Prepare the tf.data.Dataset for training
    :param data_dir: directory of the dataset
    :param batch_size: size of the batch
    :param buffer_size: the number of elements from this
        dataset from which the new dataset will sample.
    :param n_frames: size of frame rate
    :param cache: if True, cache the dataset
    :param train: if True, agument and shuffle the dataset
    :return: the dataset in input
    """
    autotune = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.list_files(os.path.join(data_dir, '*'))
    ds = ds.map(lambda x: load_frames(x, n_frames, train), num_parallel_calls=autotune)
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory. It cause memory leak, check with more memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if train:
        ds = ds.shuffle(buffer_size=buffer_size).repeat()
    
    ds = ds.batch(batch_size).prefetch(autotune)
    return ds


def load_frames(folder_path: str, n_frames: int, train: bool):
    """
    Load the frames in the folder specified by folder_path
    :param folder_path: folder path where frames are located
    :param n_frames: frame rate
    :param train: if true, augment images
    :return: the decoded frames
    """
    files = tf.io.matching_files(folder_path + "/*.jpg")
    start = tf.random.uniform([], maxval=len(files) - n_frames, dtype=tf.int32)
    sampled_indices = tf.random.shuffle(tf.range(start, start + n_frames))[:3]
    flip_sequence = tf.random.uniform([], maxval=2, dtype=tf.int32)
    sampled_indices = tf.where(
        flip_sequence == 1 and train,
        tf.sort(sampled_indices, direction="DESCENDING"),
        tf.sort(sampled_indices)
    )
    sampled_files = tf.gather(files, sampled_indices)

    frame_0 = decode_img(sampled_files[0])
    frame_1 = decode_img(sampled_files[2])
    frame_t = decode_img(sampled_files[1])
    
    if train:
        frames = data_augment(tf.concat([frame_0, frame_1, frame_t], axis=-1))
        frame_0, frame_1, frame_t = frames[:, :, :3], frames[:, :, 3:6], frames[:, :, 6:9]

    indice_t = tf.cast(tf.abs(sampled_indices[1] - sampled_indices[0]), tf.float32) \
        / tf.cast(tf.abs(sampled_indices[2] - sampled_indices[0]), tf.float32)
    return (frame_0, frame_1, indice_t, frame_t), frame_t


def data_augment(image):
    """
    Augment the image by resizing, random cropping and random flipping it
    :param image: the image to augment
    :return: the image augmented
    """
    scale = tf.random.uniform([], minval=0.8, maxval=1.0)
    w = tf.cast(W * scale, tf.int32)
    h = tf.cast(H * scale, tf.int32)
    # resize and rancom crop
    image = tf.image.random_crop(image, size=(h, w, 9))
    image = tf.image.resize(image, size=(H, W))
    # random flip
    image = tf.image.random_flip_left_right(image)
    return image


def decode_img(image: str):
    """
    Decode the image from its filename
    :param image: the image to decode
    :return: the image decoded
    """
    img_bytes = tf.io.read_file(image)
    # convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_image(img_bytes, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    #image = tf.image.resize(image, [H, W])
    return image


if __name__ == '__main__':
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    dataset = load_dataset('dataset/train', train=True, cache=False)
    for (d1, d2, d3, d4), d4 in dataset:
        print(d1.shape)

    # root = 'dataset/train'
    # images = []
    # for folder in os.listdir(root):
    #     folder_path = os.path.join(root, folder)
    #     for file in os.listdir(folder_path):
    #         path = os.path.join(folder_path, file)
    #         images.append(path)

    # ds = tf.data.Dataset.from_tensor_slices(images).map(decode_img, 8).batch(32)
    # for i, x in enumerate(ds):
    #     print(i, x.shape)