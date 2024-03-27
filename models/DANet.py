import tensorflow as tf
from Attention_modules import PAM_Module, CAM_Module

######################################################################################################
#                                                                                                    #
# ------------------------------------------- DANet -------------------------------------------------#
#                                                                                                    #
######################################################################################################
# https://github.com/junfu1115/DANet
r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
<https://arxiv.org/abs/1809.02983.pdf>`
"""

class DANet():
    def __init__(self, nclass, backbone='resnet101', input_shape=(512, 512, 3)):
        self.input_shape = input_shape
        self.pretrained = "imagenet"
        self.inputs = tf.keras.layers.Input(shape=self.input_shape)
        self.backbone = backbone

        # resnet101 and resnet50
        height, width, in_channels = (32, 32, 2048)

        inter_channels = in_channels // 4
        # Convolution layers for feature extraction
        self.conv5a = tf.keras.Sequential([
            tf.keras.layers.Conv2D(inter_channels, 3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        self.conv5c = tf.keras.Sequential([
            tf.keras.layers.Conv2D(inter_channels, 3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        in_dim = [height, width, inter_channels]
        # Position Attention Module (PAM) and Channel Attention Module (CAM)
        self.sa = PAM_Module(in_dim)
        self.sc = CAM_Module(in_dim)

        # Additional convolution layers
        self.conv51 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(inter_channels, 3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        self.conv52 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(inter_channels, 3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        # Dropout and final convolution layer
        self.conv8 = tf.keras.Sequential([
            # Dropout(0.1),
            tf.keras.layers.Conv2D(nclass, 1)
        ])

        self.resize = tf.keras.layers.Resizing(self.input_shape[0], self.input_shape[1], interpolation="bilinear")
        self.relu = tf.keras.layers.Activation('relu')


    # ResNet Bottleneck Block
    def _bottleneck_resblock(self, x, filters, stride, dilation_factor, name, identity_connection=True):
        assert filters % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name='%s_1_conv'%name)(x)
            o_b1 = tf.keras.layers.BatchNormalization(name='%s_bn'%name)(o_b1)
        else:
            o_b1 = x
        # branch2
        o_b2a = tf.keras.layers.Conv2D(filters//4, kernel_size=1, strides=1, padding='same', name='%s_2_conv'%name)(x)
        o_b2a = tf.keras.layers.BatchNormalization(name='%s_2_bn'%name)(o_b2a)
        o_b2a = tf.keras.layers.Activation("relu", name='%s_2_relu'%name)(o_b2a)

        o_b2b = tf.keras.layers.Conv2D(filters//4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name='%s_3_conv'%name)(o_b2a)
        o_b2b = tf.keras.layers.BatchNormalization(name='%s_3_bn'%name)(o_b2b)
        o_b2b = tf.keras.layers.Activation("relu", name='%s_3_relu'%name)(o_b2b)

        o_b2c = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name='%s_0_conv'%name)(o_b2b)
        o_b2c = tf.keras.layers.BatchNormalization(name='%s_0_bn'%name)(o_b2c)

        # add
        outputs = tf.keras.layers.Add(name='%s_add'%name)([o_b1, o_b2c])
        # relu
        outputs = tf.keras.layers.Activation("relu", name='%s_out'%name)(outputs)
        return outputs

    def __call__(self):
        # Load backbone
        # resnet101v2
        if self.backbone == 'resnet101':
            # backbone_model = tf.keras.applications.ResNet101V2(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            backbone_model = tf.keras.applications.ResNet101(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            third_block_layer = backbone_model.get_layer(name="conv4_block23_out").output # 32, 32, 1024
        # resnet50v2
        elif self.backbone == 'resnet50':
            # backbone_model = tf.keras.applications.ResNet50V2(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            backbone_model = tf.keras.applications.ResNet50(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            third_block_layer = backbone_model.get_layer(name="conv4_block6_out").output # 32, 32, 1024

        # x = backbone_model.get_layer(index=-1).output
        #block4
        block4 = self._bottleneck_resblock(third_block_layer, 2048, 1, 2, 'conv5_block1', identity_connection=False)
        for i in range(2, 4):
            block4 = self._bottleneck_resblock(block4, 2048, 1, 1, 'conv5_block%d'%i)
        x = block4

        # Pass through DANetHead
        # x = self.head(x)
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # Combine features from PAM and CAM
        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)

        # Resize the output
        outputs = self.resize(sasc_output)

        # Create the DANet model
        model = tf.keras.Model(inputs=self.inputs, outputs=outputs, name='DANet')
        return model
