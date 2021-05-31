from model import SloMo, BackWarp, Preprocess
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import types
import cv2
import os


class Engine(object):
    """ Base Engine Class """
    def __init__(self, config):
        self.config = config
        self.slomo = SloMo(negative_slope=config.get('negative_slope', 0.1))
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def load_weights(self, path):
        self.slomo.load_weights(path)


class TrainEngine(Engine):
    """ Train Engine Class """
    def build_model(self):
        frame_0 = layers.Input(shape=(None, None, 3), dtype=tf.float32)
        frame_1 = layers.Input(shape=(None, None, 3), dtype=tf.float32)
        t_indice = layers.Input(shape=tf.TensorShape([]), dtype=tf.float32)
        frame_true = layers.Input(shape=(None, None, 3), dtype=tf.float32)
        
        frame_pred, loss_outputs = self.slomo([frame_0, frame_1, t_indice])
        model = tf.keras.Model(
            inputs=[frame_0, frame_1, t_indice, frame_true],
            outputs=frame_pred)

        mae = tf.keras.losses.MAE
        mse = tf.keras.losses.MSE
        l1 = layers.Lambda(lambda x: tf.reduce_mean(tf.abs(x)))

        # Reconstruction loss
        loss_r = mae(frame_true, frame_pred)

        # Perceptual loss
        percept = self.build_percept()
        y_true = percept(frame_true)
        y_pred = percept(frame_pred)
        loss_p = mse(y_true, y_pred)
        
        # Warping loss
        backwarp = BackWarp()
        flow_01, flow_10, f_t0_, f_t1_ = loss_outputs
        loss_w = mae(frame_0, backwarp([frame_1, flow_01])) + \
                 mae(frame_1, backwarp([frame_0, flow_10])) + \
                 mae(frame_true, backwarp([frame_0, f_t0_])) + \
                 mae(frame_true, backwarp([frame_1, f_t1_]))
                
        # Smoothness loss
        loss_s = l1(flow_01[:, 1:, :, :] - flow_01[:, :-1, :, :]) + \
                 l1(flow_01[:, :, 1:, :] - flow_01[:, :, :-1, :]) + \
                 l1(flow_10[:, 1:, :, :] - flow_10[:, :-1, :, :]) + \
                 l1(flow_10[:, :, 1:, :] - flow_10[:, :, :-1, :])

        # Totle loss
        loss = self.config.get('rec_loss', 0.8) * 255 * tf.reduce_mean(loss_r) + \
               self.config.get('percep_loss', 0.005) * tf.reduce_mean(loss_p) + \
               self.config.get('warp_loss', 0.4) * 255 * tf.reduce_mean(loss_w) + \
               self.config.get('smooth_loss', 1.0) * loss_s
        model.add_loss(loss)

        # Add metrics
        psnr = layers.Lambda(
            lambda args: tf.image.psnr(args[0], args[1], 1.0)
        )([frame_true, frame_pred])
        ssim = layers.Lambda(
            lambda args: tf.image.ssim(args[0], args[1], 1.0)
        )([frame_true, frame_pred])
        model.add_metric(psnr, name='psnr', aggregation='mean')
        model.add_metric(ssim, name='ssim', aggregation='mean')
        
        return model

    def train(self,
              train_dataset,
              valid_dataset,
              save_path,
              learning_rate=1e-4,
              steps_per_epoch=100,
              epochs=100):
        
        if os.path.isdir(save_path):
            if not os.path.exists(save_path): os.makedirs(save_path)
            model_path = os.path.join(save_path, 'best.h5')
        else:
            model_path = save_path
            save_path = os.path.split(save_path)

        def set_model(cls, model):
            cls.model = self.slomo
            
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='psnr',
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        checkpoint.set_model = types.MethodType(set_model, checkpoint)
        
        opt = tf.keras.optimizers.Adam(learning_rate)
        self.model.compile(optimizer=opt)
        self.model.summary()
        history = self.model.fit(
            x=train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=valid_dataset,
            callbacks=[checkpoint]
        ).history

        for epoch in range(epochs):
            if history.get(checkpoint.monitor)[epoch] == checkpoint.best:
                loss = history['val_loss'][epoch]
                psnr = history['val_psnr'][epoch]
                ssim = history['val_ssim'][epoch]
                model_name = 'slomo_loss:{:.4}_psnr:{:.4}_ssim:{:.4}.h5'.format(
                    loss, psnr, ssim)
                os.rename(model_path, os.path.join(save_path, model_name))
                break

    def build_percept(self):
        vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
        vgg16_b43 = tf.keras.Model(
            vgg16.inputs, vgg16.get_layer("block4_conv3").output, trainable=False
        )
        percept = tf.keras.Sequential([Preprocess(), vgg16_b43])
        return percept


class InferEngine(Engine):
    """ Infer Engine Class """
    def build_model(self):
        frame_0 = layers.Input(shape=(None, None, 3), dtype=tf.float32)
        frame_1 = layers.Input(shape=(None, None, 3), dtype=tf.float32)
        t_indice = layers.Input(shape=tf.TensorShape([]), dtype=tf.float32)
        frame_pred, _ = self.slomo([frame_0, frame_1, t_indice])
        model = tf.keras.Model(inputs=[frame_0, frame_1, t_indice], outputs=frame_pred)
        return model

    def interpolate(self, src, tgt, tgt_fps,
                    batch_size=32, fourcc='mp4v',
                    tgt_weidth=None, tgt_height=None):
        
        capture = cv2.VideoCapture(src)
        if not capture.isOpened():
            raise Exception("{} open failed!".format(src))

        src_fps = int(capture.get(cv2.CAP_PROP_FPS))
        src_weidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


        if tgt_weidth is None or tgt_height is None:
            tgt_weidth = src_weidth
            tgt_height = src_height

        # 插多少帧
        interp_frames = round(tgt_fps / src_fps) - 1

        def _gen():
            while True:
                ret, frame = capture.read()
                if not ret: break
                frame = np.asarray(frame, np.float32) / 255
                yield frame

        autotune = tf.data.experimental.AUTOTUNE
        ds = tf.data.Dataset.from_generator(
            _gen, output_types=tf.float32,
            output_shapes=(src_height, src_weidth, 3)
        )
        if tgt_weidth < src_weidth and tgt_height < src_height:
            ds = ds.map(lambda x: tf.image.resize(x, (tgt_height, tgt_weidth)), autotune)
            
        ds = ds.batch(batch_size).prefetch(batch_size)

        fourcc = cv2.VideoWriter_fourcc(*fourcc.upper())
        writer = cv2.VideoWriter(tgt, fourcc, tgt_fps, (tgt_height, tgt_weidth))
        for frames in ds:
            frames_interp = self.slomo.interpolate(frames, interp_frames)
            frames_interp = np.asarray(frames_interp.numpy() * 255, np.uint8)
            for frame in frames_interp:
                writer.write(frame)

        capture.release()
        writer.release()
        

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    #tf.compat.v1.disable_eager_execution()

    engine = TrainEngine({})
    engine.model.summary()

    from data import load_dataset
    dataset = load_dataset('dataset/train')
    for xs in dataset:
        engine.model(xs)