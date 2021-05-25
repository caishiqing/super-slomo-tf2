from model import SloMo, BackWarp, Preprocess
from tensorflow.keras import layers
import tensorflow as tf
import os


class Engine(object):
    """ Base Engine Class """
    def __init__(self, config):
        self.config = config
        self.slomo = SloMo(
            negative_slope=config.get('negative_slope', 0.1))
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError


class TrainEngine(Engine):
    """ Train Engine Class """
    def __init__(self, config):
        super(TrainEngine, self).__init__(config)

    def train(self,
              train_dataset,
              valid_dataset,
              save_path,
              learning_rate=1e-4,
              steps_per_epoch=100,
              epochs=100):
        
        if os.path.isdir(save_path):
            if not os.path.exists(save_path): os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.h5')
            
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='psnr',
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        opt = tf.keras.optimizers.Adam(learning_rate)
        self.model.compile(optimizer=opt)
        self.model.summary()
        self.model.fit(
            x=train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=valid_dataset,
            callbacks=[checkpoint]
        )

    # @tf.function
    # def train_step(self, data, optimizer, strategy=None):
    #     def _step_fn(data):
    #         frame_0, frame_1, indice_t, frame_t = data

    #         with tf.GradientTape() as tape:
    #             frame_pred, loss_outputs = self.slomo([frame_0, frame_1, indice_t])

    #             mae = tf.keras.losses.MAE
    #             mse = tf.keras.losses.MSE
    #             l1 = layers.Lambda(lambda x: tf.reduce_mean(tf.abs(x)))

    #             # Reconstruction loss
    #             loss_r = mae(frame_t, frame_pred)

    #             # Perceptual loss
    #             percept = self.build_percept()
    #             y_true = percept(frame_t)
    #             y_pred = percept(frame_pred)
    #             loss_p = mse(y_true, y_pred)
                
    #             # Warping loss
    #             backwarp = BackWarp()
    #             flow_01, flow_10, f_t0_, f_t1_ = loss_outputs
    #             loss_w = mae(frame_0, backwarp([frame_1, flow_01])) + \
    #                     mae(frame_1, backwarp([frame_0, flow_10])) + \
    #                     mae(frame_t, backwarp([frame_0, f_t0_])) + \
    #                     mae(frame_t, backwarp([frame_1, f_t1_]))
                        
    #             # Smoothness loss
    #             loss_s = l1(flow_01[:, 1:, :, :] - flow_01[:, :-1, :, :]) + \
    #                     l1(flow_01[:, :, 1:, :] - flow_01[:, :, :-1, :]) + \
    #                     l1(flow_10[:, 1:, :, :] - flow_10[:, :-1, :, :]) + \
    #                     l1(flow_10[:, :, 1:, :] - flow_10[:, :, :-1, :])

    #             # Totle loss
    #             loss = self.config.get('rec_loss', 0.8) * 255 * tf.reduce_mean(loss_r) + \
    #                 self.config.get('percep_loss', 0.005) * tf.reduce_mean(loss_p) + \
    #                 self.config.get('warp_loss', 0.4) * 255 * tf.reduce_mean(loss_w) + \
    #                 self.config.get('smooth_loss', 1.0) * loss_s

    #             # Add metrics
    #             psnr = tf.image.psnr(frame_t, frame_pred, 1.0)
    #             ssim = tf.image.ssim(frame_t, frame_pred, 1.0)

    #         variables = self.slomo.trainable_variables
    #         gradients = tape.gradient(loss, variables)
    #         optimizer.apply_gradients(zip(gradients, variables))

    #         return loss, psnr, ssim
            
    #     if strategy is None:
    #         return _step_fn(data)

    #     with strategy.scope():
    #         loss, psnr, ssim = self.strategy.experimental_run_v2(_step_fn, args=(data,))

    #     return (self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None),
    #             self.strategy.reduce(tf.distribute.ReduceOp.MEAN, psnr, axis=None),
    #             self.strategy.reduce(tf.distribute.ReduceOp.MEAN, ssim, axis=None))

    # @tf.function
    # def test_step(self, data, strategy=None):
    #     def _step_fn(data):
    #         frame_0, frame_1, indice_t, frame_t = data
    #         frame_pred, loss_outputs = self.slomo([frame_0, frame_1, indice_t])

    #         mae = tf.keras.losses.MAE
    #         mse = tf.keras.losses.MSE
    #         l1 = layers.Lambda(lambda x: tf.reduce_mean(tf.abs(x)))

    #         # Reconstruction loss
    #         loss_r = mae(frame_t, frame_pred)

    #         # Perceptual loss
    #         percept = self.build_percept()
    #         y_true = percept(frame_t)
    #         y_pred = percept(frame_pred)
    #         loss_p = mse(y_true, y_pred)
            
    #         # Warping loss
    #         backwarp = BackWarp()
    #         flow_01, flow_10, f_t0_, f_t1_ = loss_outputs
    #         loss_w = mae(frame_0, backwarp([frame_1, flow_01])) + \
    #                  mae(frame_1, backwarp([frame_0, flow_10])) + \
    #                  mae(frame_t, backwarp([frame_0, f_t0_])) + \
    #                  mae(frame_t, backwarp([frame_1, f_t1_]))
                    
    #         # Smoothness loss
    #         loss_s = l1(flow_01[:, 1:, :, :] - flow_01[:, :-1, :, :]) + \
    #                  l1(flow_01[:, :, 1:, :] - flow_01[:, :, :-1, :]) + \
    #                  l1(flow_10[:, 1:, :, :] - flow_10[:, :-1, :, :]) + \
    #                  l1(flow_10[:, :, 1:, :] - flow_10[:, :, :-1, :])

    #         # Totle loss
    #         loss = self.config.get('rec_loss', 0.8) * 255 * tf.reduce_mean(loss_r) + \
    #                self.config.get('percep_loss', 0.005) * tf.reduce_mean(loss_p) + \
    #                self.config.get('warp_loss', 0.4) * 255 * tf.reduce_mean(loss_w) + \
    #                self.config.get('smooth_loss', 1.0) * loss_s

    #         # Add metrics
    #         psnr = tf.image.psnr(frame_t, frame_pred, 1.0)
    #         ssim = tf.image.ssim(frame_t, frame_pred, 1.0)

    #         return loss, psnr, ssim
            
    #     if strategy is None:
    #         return _step_fn(data)

    #     with strategy.scope():
    #         loss, psnr, ssim = self.strategy.experimental_run_v2(_step_fn, args=(data,))

    #     return (self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None),
    #             self.strategy.reduce(tf.distribute.ReduceOp.MEAN, psnr, axis=None),
    #             self.strategy.reduce(tf.distribute.ReduceOp.MEAN, ssim, axis=None))

        
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

    def build_percept(self):
        vgg16 = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
        vgg16_b43 = tf.keras.Model(
            vgg16.inputs, vgg16.get_layer("block4_conv3").output, trainable=False
        )
        percept = tf.keras.Sequential([Preprocess(), vgg16_b43])
        return percept


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