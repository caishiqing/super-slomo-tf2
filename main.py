from engine import TrainEngine, InferEngine
from data import load_dataset
import tensorflow as tf
import random
import shutil
import yaml
import fire
import os

config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)


def process_video(video_dir: str,
                  output_dir: str,
                  name_len: int = 4,
                  test_size: float = 0.1): 
    """将视频文件预处理成图片帧

    Args:
        video_dir (str): 存放所有视频文件的目录
        output_dir (str): 输出数据集路径
        name_len (int, optional): 文件名长度
        test_size (float, optional): 测试集划分比例
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    videos = os.listdir(video_dir)
    random.shuffle(videos)
    test_num = int(len(videos) * test_size)
    test_videos = videos[:test_num]
    train_videos = videos[test_num:]

    def _process(videos, output_dir):
        for video in videos:
            video_name = os.path.splitext(video)[0]
            video_path = os.path.join(video_dir, video)
            target_dir = os.path.join(output_dir, video_name)
            os.makedirs(target_dir)
            ret = os.system(
                "ffmpeg -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%0{}d.jpg".format(
                    video_path, config['width'], config['height'], target_dir, name_len)
            )
            if ret: print("Error converting file:{}. Exiting.".format(video))
            if len(os.listdir(target_dir)) < config['n_frames']:
                shutil.rmtree(target_dir)

    _process(train_videos, os.path.join(output_dir, 'train'))
    _process(test_videos, os.path.join(output_dir, 'test'))


def train(data_path: str,
          batch_size: int = 8,
          epochs: int = 20,
          steps_per_epoch: int = 100,
          learning_rate: float = 1e-4,
          save_path: int = 'models',
          gpu=0):
    """ 训练

    Args:
        data_path (str): 训练集目录，其中必须包含 'train' 和 'test' 两个子目录
        batch_size (int, optional): 批大小
        epochs (int, optional): 训练轮数
        steps_per_epoch (int, optional): 每轮迭代步数
        learning_rate (float, optional): 学习率
        save_path (int, optional): 模型保存路径
        gpu (str, optional): GPU设备ID
    """
    if isinstance(gpu, int): gpu = str(gpu)
    elif isinstance(gpu, tuple): gpu = ','.join(map(str, gpu))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    strategy = tf.distribute.MirroredStrategy()
    n_gpus = strategy.num_replicas_in_sync
    batch_size *= max(n_gpus, 1)

    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    train_dataset = load_dataset(
        train_path,
        batch_size=batch_size,
        n_frames=config['n_frames'],
        cache=True,
        train=True,
    )
    valid_dataset = load_dataset(
        valid_path,
        batch_size=batch_size,
        n_frames=config['n_frames'],
        cache=True,
        train=False,
    )
    with strategy.scope(): 
        engine = TrainEngine(config)
        engine.train(
            train_dataset,
            valid_dataset,
            save_path,
            learning_rate=learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )


def interpolate(model_path: str, 
                src_file: str,
                tgt_file: str,
                tgt_fps: int,
                tgt_weidth: int = None,
                tgt_height: int = None, 
                batch_size: int = 8,
                fourcc: str = 'xvid',
                gpu: str = ''):
    """[summary]

    Args:
        model_path (str): [description]
        src_file (str): [description]
        tgt_file (str): [description]
        tgt_fps (int): [description]
        tgt_weidth (int, optional): [description]. Defaults to None.
        tgt_height (int, optional): [description]. Defaults to None.
        batch_size (int, optional): [description]. Defaults to 32.
        fourcc (str, optional): [description]. Defaults to 'x264'.
        gpu (str, optional): [description]. Defaults to ''.
    """
    if isinstance(gpu, int): gpu = str(gpu)
    elif isinstance(gpu, tuple): gpu = ','.join(map(str, gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

    engine = InferEngine(config)
    engine.load_weights(model_path)
        
    engine.interpolate(
        src_file, tgt_file, tgt_fps,
        batch_size=batch_size,
        fourcc=fourcc,
        tgt_weidth=tgt_weidth,
        tgt_height=tgt_height
    )
    


if __name__ == '__main__':
    fire.Fire()
        