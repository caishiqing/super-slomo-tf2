from tensorflow.python.ops import image_ops, array_ops
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
        self.conv1 = layers.Conv2D(
            self.filters, self.kernel_size, padding='same')
        self.conv2 = layers.Conv2D(
            self.filters, self.kernel_size, padding='same')
        
    def call(self, inputs):
        x, skpCn = inputs
        new_shape = array_ops.shape(skpCn)[1:3]
        # Upsample(resize) to skpCn
        x = image_ops.resize_images_v2(
            x, size=new_shape,
            method=image_ops.ResizeMethod.BILINEAR,
            name="upsampling")

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
        frame_head.set_shape(frame_tail.shape)
        return frame_head

    def compute_output_shape(self, input_shape):
        frame_shape = input_shape[0]
        return frame_shape


class FlowOptical(layers.Layer):
    """ Flow Optical Compute """
    def __init__(self, negative_slope, name="flow_optical", **kwargs):
        super(FlowOptical, self).__init__(name=name, **kwargs)
        self.negative_slope = negative_slope
        
    def build(self, input_shape):
        self.flow_comp = UNet(4, negative_slope=self.negative_slope, name="flow_comp")

    def call(self, inputs):
        frame_0, frame_1 = inputs

        # Compute flow
        flow_input = tf.concat([frame_0, frame_1], axis=-1)
        flow_output = self.flow_comp(flow_input)
        flow_01 = flow_output[:, :, :, :2]
        flow_10 = flow_output[:, :, :, 2:]
        
        return flow_01, flow_10

    def get_config(self):
        config = super(FlowOptical, self).get_config()
        config['negative_slope'] = self.negative_slope
        

class FlowInterp(layers.Layer):
    """ Flow Interpolation """
    def __init__(self, negative_slope, name="flow_interp", **kwargs):
        super(FlowInterp, self).__init__(name=name, **kwargs)
        self.negative_slope = negative_slope
        
    def build(self, input_shape):
        self.flow_interp = UNet(5, negative_slope=self.negative_slope, name="flow_interp")
        self.backwarp = BackWarp()

    def call(self, inputs):
        frame_0, frame_1, t_indices, flow_01, flow_10 = inputs

        f_t0_ = -(1 - t_indices) * t_indices * flow_01 + t_indices * t_indices * flow_10
        f_t1_ = (1 - t_indices) * (1 - t_indices) * flow_01 - t_indices * (1 - t_indices) * flow_10
        
        # Intermediate Frame Synthesis
        g_t0_ = self.backwarp([frame_0, f_t0_])
        g_t1_ = self.backwarp([frame_1, f_t1_])
        
        # Arbitrary-time Flow Interpolation
        flow_interp_in = tf.concat(
            [frame_0, frame_1, flow_01, flow_10, f_t0_, f_t1_, g_t0_, g_t1_],
            axis=-1
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

        return f_t0_, f_t1_, v_t0, v_t1, g_t0, g_t1

    def get_config(self):
        config = super(FlowOptical, self).get_config()
        config['negative_slope'] = self.negative_slope


class FrameSynthesis(layers.Layer):
    """ Intermediate Frame Synthesis """
    def __init__(self, name="frame_synsesis", **kwargs):
        super(FrameSynthesis, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        t_indices, v_t0, v_t1, g_t0, g_t1 = inputs
        
        # Synthesize the intermediate image
        z = (1 - t_indices) * v_t0 + t_indices * v_t1
        fram_pred = 1 / z * ((1 - t_indices) * v_t0 * g_t0 + t_indices * v_t1 * g_t1)
        return fram_pred


class SloMo(tf.keras.Model):
    """ Super-SloMo Model """
    def __init__(self, negative_slope=0.1, name="slomo", **kwargs):
        super(SloMo, self).__init__(name=name, **kwargs)
        self.negative_slope = negative_slope
        self.flow_optical = FlowOptical(negative_slope, name="flow_optical")
        self.flow_interp = FlowInterp(negative_slope, name="flow_interp")
        self.frame_synthesis = FrameSynthesis(name="frame_synthesis")
        
    def call(self, inputs):
        frame_0, frame_1, t_indice = inputs
        t_indice = t_indice[:, tf.newaxis, tf.newaxis, tf.newaxis]

        flow_01, flow_10 = self.flow_optical([frame_0, frame_1])
        f_t0_, f_t1_, v_t0, v_t1, g_t0, g_t1 = self.flow_interp(
            [frame_0, frame_1, t_indice, flow_01, flow_10]
        )
        frame_pred = self.frame_synthesis([t_indice, v_t0, v_t1, g_t0, g_t1])
        
        loss_outputs = [flow_01, flow_10, f_t0_, f_t1_]
        return frame_pred, loss_outputs

    def interpolate(self, frames, interp_frames=1):
        frame_0, frame_1 = frames[:-1], frames[1:]
        flow_01, flow_10 = self.flow_optical([frame_0, frame_1])
        t_indice = tf.cast(tf.linspace(0, 1, interp_frames + 2)[1:-1], tf.float32)
        t_indice = tf.tile(t_indice[tf.newaxis, :], [array_ops.shape(frame_0)[0], 1])
        t_indice = tf.reshape(t_indice, [-1])
        t_indice = t_indice[:, tf.newaxis, tf.newaxis, tf.newaxis]
        
        # batch interpolate
        b, w, h, c = array_ops.shape(frame_0)

        frame_0 = tf.expand_dims(frame_0, 1)
        frame_0 = tf.tile(frame_0, [1, interp_frames, 1, 1, 1])
        frame_0 = tf.reshape(frame_0, [-1, w, h, c])

        frame_1 = tf.expand_dims(frame_1, 1)
        frame_1 = tf.tile(frame_1, [1, interp_frames, 1, 1, 1])
        frame_1 = tf.reshape(frame_1, [-1, w, h, c])

        flow_01 = tf.expand_dims(flow_01, 1)
        flow_01 = tf.tile(flow_01, [1, interp_frames, 1, 1, 1])
        flow_01 = tf.reshape(flow_01, [-1, w, h, 2])

        flow_10 = tf.expand_dims(flow_10, 1)
        flow_10 = tf.tile(flow_10, [1, interp_frames, 1, 1, 1])
        flow_10 = tf.reshape(flow_10, [-1, w, h, 2])
        
        f_t0_, f_t1_, v_t0, v_t1, g_t0, g_t1 = self.flow_interp(
            [frame_0, frame_1, t_indice, flow_01, flow_10]
        )
        frame_interp = self.frame_synthesis([t_indice, v_t0, v_t1, g_t0, g_t1])

        frame_0 = tf.expand_dims(frames[:-1], 1)
        frame_interp = tf.reshape(frame_interp, [-1, interp_frames, w, h, c])
        frames = tf.concat([frame_0, frame_interp], axis=1)
        frames = tf.reshape(frames, [-1, w, h, c])
        
        return frames

    def get_config(self):
        config = super(SloMo, self).get_config()
        config['negative_slope'] = self.negative_slope
        return config


class Preprocess(layers.Layer):
    """ Normalize image """
    def __init__(self, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225], **kwargs):
        super(Preprocess, self).__init__(**kwargs)
        self.mean = tf.constant(mean, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :]
        self.std = tf.constant(std, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :]
        
    def call(self, inputs):
        return (inputs - self.mean) / self.std
        
    
tf.keras.utils.get_custom_objects().update(
    {
        'Encoder': Encoder,
        'Decoder': Decoder,
        'UNet': UNet,
        'BackWarp': BackWarp,
        'FlowOptical': FlowOptical,
        'FlowInterp': FlowInterp,
        'FrameSynthesis': FrameSynthesis,
        'SloMo': SloMo, 
    }
)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    slomo = SloMo()
    frames = tf.random.uniform((10, 224, 224, 3), dtype=tf.float32)
    frames = slomo.interpolate(frames, 8)
    print(frames.shape)