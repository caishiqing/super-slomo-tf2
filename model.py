from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf


class Encoder(layers.Layer):
    """
    A class for creating neural network blocks containing layers:
    
    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.
    """
    def __init__(self, filters,
                 kernel_size=3,
                 pool_size=2,
                 negative_slope=0.1,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.negative_slope = negative_slope

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            self.filters, self.kernel_size,
            padding='same')
        self.conv2 = layers.Conv2D(
            self.filters, self.kernel_size,
            padding='same')
                
    def call(self, inputs):
        x = layers.AveragePooling2D(self.pool_size)(inputs)
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, self.negative_slope)
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x, self.negative_slope)
        return x

    def get_config(self):
        config = super(Encoder, self).get_config()
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['pool_size'] = self.pool_size
        config['negative_slope'] = self.negative_slope
        return config


class Decoder(layers.Layer):
    """
    A class for creating neural network blocks containing layers:
    
    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.
    """
    def __init__(self, filters,
                 kernel_size=3,
                 scale_factor=2,
                 negative_slope=0.1,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.negative_slope = negative_slope

    def build(self, input_shape):
        self.interpolation = layers.UpSampling2D(
            size=(self.scale_factor, self.scale_factor),
            interpolation="bilinear")
        self.conv1 = layers.Conv2D(
            self.filters, self.kernel_size,
            padding='same')
        self.conv2 = layers.Conv2D(
            self.filters, self.kernel_size,
            padding='same')
        
    def call(self, inputs):
        x, skpCn = inputs
        x = self.interpolation(x)
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, self.negative_slope)
        x = tf.concat([x, skpCn], axis=3)
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x, self.negative_slope)
        return x

    def get_config(self):
        config = super(Decoder, self).get_config()
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['scale_factor'] = self.scale_factor
        config['negative_slope'] = self.negative_slope
        return config


class UNet(layers.Layer):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    """
    def __init__(self, units, negative_slope=0.1, **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.units = units
        self.negative_slope = negative_slope

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(32, 7, padding='same')
        self.conv2 = layers.Conv2D(32, 7, padding='same')
        self.down1 = Encoder(64, 5, 2, negative_slope=self.negative_slope)
        self.down2 = Encoder(128, 3, 2, negative_slope=self.negative_slope)
        self.down3 = Encoder(256, 3, 2, negative_slope=self.negative_slope)
        self.down4 = Encoder(512, 3, 2, negative_slope=self.negative_slope)
        self.down5 = Encoder(512, 3, 2, negative_slope=self.negative_slope)
        self.up1 = Decoder(512, 3, 2, negative_slope=self.negative_slope)
        self.up2 = Decoder(256, 3, 2, negative_slope=self.negative_slope)
        self.up3 = Decoder(128, 3, 2, negative_slope=self.negative_slope)
        self.up4 = Decoder(64, 3, 2, negative_slope=self.negative_slope)
        self.up5 = Decoder(32, 3, 2, negative_slope=self.negative_slope)
        self.conv3 = layers.Conv2D(self.units, 3, padding='same')

    def call(self, inputs):
        x = tf.nn.leaky_relu(self.conv1(inputs), self.negative_slope)
        s1 = tf.nn.leaky_relu(self.conv2(x), self.negative_slope)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1([x, s5])
        x = self.up2([x, s4])
        x = self.up3([x, s3])
        x = self.up4([x, s2])
        x = self.up5([x, s1])
        x = tf.nn.leaky_relu(self.conv3(x), self.negative_slope)
        return x

    def get_config(self):
        config = super(UNet, self).get_config()
        config['units'] = self.units
        config['negative_slope'] = self.negative_slope
        return config


class BackWarp(layers.Layer):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    """
    def call(self, inputs):
        frame_tail, flow = inputs
        frame_head = tfa.image.dense_image_warp(frame_tail, flow)
        return frame_head


class FramSynthesis(layers.Layer):
    """ Intermediate Frame Synthesis """
    def __init__(self, negative_slope, name="frame_synsesis", **kwargs):
        super(FramSynthesis, self).__init__(name=name, **kwargs)
        self.negative_slope = negative_slope
        
    def build(self, input_shape):
        self.flow_comp = UNet(4, negative_slope=self.negative_slope, name="flow_comp")
        self.flow_interp = UNet(5, negative_slope=self.negative_slope, name="flow_interp")
        self.backwarp = BackWarp()
        
    def call(self, inputs):
        frame_0, frame_1, t_indices = inputs  # t_indices shape (batch,)
        t_indices = t_indices[:, tf.newaxis, tf.newaxis, tf.newaxis]

        # Compute flow
        flow_input = tf.concat([frame_0, frame_1], axis=-1)
        flow_output = self.flow_comp(flow_input)
        flow_01 = flow_output[:, :, :, :2]
        flow_10 = flow_output[:, :, :, 2:]
        
        # Optical Flow
        f_t0_ = -(1 - t_indices) * t_indices * flow_01 + t_indices * t_indices * flow_10
        f_t1_ = (1 - t_indices) * (1 - t_indices) * flow_01 - t_indices * (1 - t_indices) * flow_10
        
        # Intermediate Frame Synthesis
        g_t0_ = self.backwarp([frame_0, f_t0_])
        g_t1_ = self.backwarp([frame_1, f_t1_])
        
        # Arbitrary-time Flow Interpolation
        flow_interp_in = tf.concat(
            [frame_0, frame_1, flow_01, flow_10, f_t0_, f_t1_, g_t0_, g_t1_], axis=-1
        )
        flow_interp_out = self.flow_interp(flow_interp_in)
        
        # Optical flow residuals
        delta_flow_t0 = flow_interp_out[:, :, :, :2]
        delta_flow_t1 = flow_interp_out[:, :, :, 2:4]
        
        # Visibility maps
        v_t0 = tf.nn.sigmoid(flow_interp_out[:, :, :, 4:5])
        v_t1 = 1 - v_t0
        
        # Optical Flow estimate
        flow_t0 = f_t0_ + delta_flow_t0
        flow_t1 = f_t1_ + delta_flow_t1
        g_t0 = self.backwarp([frame_0, flow_t0])
        g_t1 = self.backwarp([frame_1, flow_t1])
        
        # Synthesize the intermediate image
        z = (1 - t_indices) * v_t0 + t_indices * v_t1
        fram_pred = 1 / z * ((1 - t_indices) * v_t0 * g_t0 + t_indices * v_t1 * g_t1)

        loss_outputs = [flow_01, flow_10, f_t0_, f_t1_]
        return fram_pred, loss_outputs

    def get_config(self):
        config = super(FramSynthesis, self).get_config()
        config['negative_slope'] = self.negative_slope
        return config


class SloMo(tf.keras.Model):
    """ Super-SloMo Model """
    def __init__(self, n_frames=12, negative_slope=0.1, name="slomo", **kwargs):
        super(SloMo, self).__init__(name=name, **kwargs)
        self.n_frames = n_frames
        self.negative_slope = negative_slope
        self.t_slices = tf.cast(tf.linspace(0, 1, self.n_frames), tf.float32)
        self.frame_synthesis = FramSynthesis(self.negative_slope)
        
    def call(self, inputs):
        frame_0, frame_1, frame_indice = inputs
        t_indices = tf.gather(self.t_slices, frame_indice)
        outputs = self.frame_synthesis([frame_0, frame_1, t_indices])
        return outputs

    def get_config(self):
        config = super(SloMo, self).get_config()
        config['n_frames'] = self.n_frames
        config['negative_slope'] = self.negative_slope
        return config
    