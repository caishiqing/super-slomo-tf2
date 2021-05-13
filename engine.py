from tensorflow.keras import layers
from model import SloMo, BackWarp
import tensorflow as tf


class Engine(object):
    """ Base Engine Class """
    def __init__(self, config):
        self.config = config
        self.slomo = SloMo(
            n_frames=config.get('n_frams', 12),
            negative_slope=config.get('negative_slope', 0.1)
        )
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError


class TrainEngine(Engine):
    """ Train Engine Class """
    def __init__(self, config):
        super(TrainEngine, self).__init__(config)
        
    def build_model(self):
        frame_0 = layers.Input(shape=(None, None, 3), dtype=tf.float32)
        frame_1 = layers.Input(shape=(None, None, 3), dtype=tf.float32)
        frame_indice = layers.Input(shape=tf.TensorShape([]), dtype=tf.int32)
        frame_true = layers.Input(shape=(None, None, 3), dtype=tf.float32)
        
        frame_pred, loss_outputs = self.slomo([frame_0, frame_1, frame_indice])
        model = tf.keras.Model(
            inputs=[frame_0, frame_1, frame_indice, frame_true],
            outputs=frame_pred)

        mae = tf.keras.losses.MAE
        mse = tf.keras.losses.MSE
        l1 = layers.Lambda(lambda x: tf.reduce_mean(tf.abs(x)))

        # Reconstruction loss
        loss_r = mae(frame_true, frame_pred)

        # Perceptual loss
        vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
        vgg16_b43 = tf.keras.Model(
            vgg16.inputs, vgg16.get_layer("block4_conv3").output, trainable=False)
        y_true = vgg16_b43(frame_true)
        y_pred = vgg16_b43(frame_pred)
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
        psnr = tf.image.psnr(frame_true, frame_pred, 1.0)
        ssim = tf.image.ssim(frame_true, frame_pred, 1.0)
        model.add_metric(psnr, name='psnr', aggregation='mean')
        model.add_metric(ssim, name='ssim', aggregation='mean')

        return model


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.compat.v1.disable_eager_execution()

    engine = TrainEngine({})
    engine.model.summary()