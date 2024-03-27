import tensorflow as tf


######################################################################################################
#                                                                                                    #
# ----------------------------------------- EAR-Net(2021) ------------------------------------------#
#                                                                                                    #
######################################################################################################

# https://github.com/synml/segmentation-pytorch/blob/main/models/ear_net.py

class ASPP_EARNet(tf.keras.layers.Layer):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP_EARNet, self).__init__()

        self.DSConv1 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(out_channels, 1, padding='same', name="aspp1_sepconv", use_bias=False),
            tf.keras.layers.BatchNormalization(name="aspp1_bn"),
            tf.keras.layers.ReLU(name="aspp1_relu")])

        self.DSConv6 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(out_channels, 1, dilation_rate=atrous_rates[0], padding='same', name="aspp6_sepconv", use_bias=False),
            tf.keras.layers.BatchNormalization(name="aspp6_bn"),
            tf.keras.layers.ReLU(name="aspp6_relu")])

        self.DSConv12 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(out_channels, 1, dilation_rate=atrous_rates[1], padding='same', name="aspp12_sepconv", use_bias=False),
            tf.keras.layers.BatchNormalization(name="aspp12_bn"),
            tf.keras.layers.ReLU(name="aspp12_relu")])

        self.DSConv18 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(out_channels, 1, dilation_rate=atrous_rates[2], padding='same', name="aspp18_sepconv", use_bias=False),
            tf.keras.layers.BatchNormalization(name="aspp18_bn"),
            tf.keras.layers.ReLU(name="aspp18_relu")])

        self.pooling = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="aspp_pool"),
            tf.keras.layers.SeparableConv2D(out_channels, 1, padding='same', name="aspp_pool_sepconv", use_bias=False),
            tf.keras.layers.BatchNormalization(name="aspp_pool_bn"),
            tf.keras.layers.ReLU(name="aspp_pool_relu")
        ])

        self.concat = tf.keras.layers.Concatenate(name="aspp_concat")
        self.aspp_last_conv = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(out_channels, 1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()])

    def call(self, x):
        res1 = self.DSConv1(x)
        res6 = self.DSConv6(x)
        res12 = self.DSConv12(x)
        res18 = self.DSConv18(x)
        pool = self.pooling(x)
        res_pool = tf.keras.layers.UpSampling2D(size=(x.shape[1]//pool.shape[1], x.shape[2]//pool.shape[2]), interpolation="bilinear", name='aspp_pool_avg')(pool)
        res = self.concat([res1, res6, res12, res18, res_pool])
        res = self.aspp_last_conv(res)
        return res

class EAR_Net(tf.keras.Model):
    def __init__(self, num_classes=16, input_shape=(512,512,3)):
        super(EAR_Net, self).__init__()
        self.input_size = input_shape
        self.inputs = tf.keras.Input(shape=self.input_size)

        # Backbone
        self.stem_block = self.make_stem_block(64)

        # ASPP
        self.aspp = ASPP_EARNet(2048, (6, 12, 18), 256)

        # Decoder
        self.compress_low_level_feature3 = self.make_compressor(64)
        self.compress_low_level_feature2 = self.make_compressor(32)
        self.compress_low_level_feature1 = self.make_compressor(16)
        self.decode3 = self.make_decoder(256)
        self.decode2 = self.make_decoder(128)
        self.decode1 = self.make_decoder(64)

        # Classifier
        self.classifier = tf.keras.layers.SeparableConv2D(num_classes, kernel_size=1)

    def _bottleneck_resblock(self, x, filters, stride, dilation_factor, name, identity_connection=True):
        assert filters % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name='%s_0_conv'%name)(x)
            o_b1 = tf.keras.layers.BatchNormalization(name='%s_0_bn'%name)(o_b1)
        else:
            o_b1 = x
        # branch2
        o_b2a = tf.keras.layers.Conv2D(filters / 4, kernel_size=1, strides=1, padding='same', name='%s_1_conv'%name)(x)
        o_b2a = tf.keras.layers.BatchNormalization(name='%s_1_bn'%name)(o_b2a)
        o_b2a = tf.keras.layers.Activation("relu", name='%s_1_relu'%name)(o_b2a)
        o_b2b = tf.keras.layers.Conv2D(filters / 4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name='%s_2_conv'%name)(o_b2a)
        o_b2b = tf.keras.layers.BatchNormalization(name='%s_2_bn'%name)(o_b2b)
        o_b2b = tf.keras.layers.Activation("relu", name='%s_2_relu'%name)(o_b2b)
        o_b2c = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name='%s_3_conv'%name)(o_b2b)
        o_b2c = tf.keras.layers.BatchNormalization(name='%s_3_bn'%name)(o_b2c)
        # add
        outputs = tf.keras.layers.Add(name='%s_add'%name)([o_b1, o_b2c])
        # relu
        outputs = tf.keras.layers.Activation("relu", name='%s_out'%name)(outputs)
        return outputs

    def build_backbone(self, inputs):
        # ResNet50
        blocks = [3, 4, 6, 3]
        resnet_strides = [1, 2, 2, 2]
        resnet_dilations = [1, 1, 1, 1]
        #block1
        block1 = self._bottleneck_resblock(inputs, 256, resnet_strides[0], resnet_dilations[0], 'conv2_block1', identity_connection=False)
        for i in range(2, blocks[0]+1):
            block1 = self._bottleneck_resblock(block1, 256, 1, 1, 'conv2_block%d'%i)
        #block2
        block2 = self._bottleneck_resblock(block1, 512, resnet_strides[1], resnet_dilations[1], 'conv3_block1', identity_connection=False)
        for i in range(2, blocks[1]+1):
            block2 = self._bottleneck_resblock(block2, 512, 1, 1, 'conv3_block%d'%i)
        #block3
        block3 = self._bottleneck_resblock(block2, 1024, resnet_strides[2], resnet_dilations[2], 'conv4_block1', identity_connection=False)
        for i in range(2, blocks[2]+1):
            block3 = self._bottleneck_resblock(block3, 1024, 1, 1, 'conv4_block%d'%i)
        #block4
        block4 = self._bottleneck_resblock(block3, 2048, resnet_strides[3], resnet_dilations[3], 'conv5_block1', identity_connection=False)
        for i in range(2, blocks[3]+1):
            block4 = self._bottleneck_resblock(block4, 2048, 1, 1, 'conv5_block%d'%i)

        model = tf.keras.Model(inputs=inputs, outputs=block4)
        weight_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(2048, 4096, 3))

        # Exclude input layer model.layers[1:]
        for l in model.layers[1:]:
            weight_layer = None
            try:
                weight_layer = weight_model.get_layer(l.name)
            except:
                print("An exception occurred with layer : ", l.name)

            if weight_layer is not None:
                l.set_weights(weight_layer.get_weights())
        return model

    def make_stem_block(self, out_channels):
        return tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(out_channels//2, kernel_size=3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def make_compressor(self, out_channels: int):
        return tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(out_channels, kernel_size=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def make_decoder(self, out_channels: int):
        return tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.SeparableConv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def __call__(self):
        # Encoder
        x = self.inputs
        x = self.stem_block(x)

        self.backbone = self.build_backbone(inputs = x)
        # self.backbone(x)
        encode1 = self.backbone.get_layer('conv2_block3_out').output
        encode2 = self.backbone.get_layer('conv3_block4_out').output
        encode3 = self.backbone.get_layer('conv4_block6_out').output
        encode4 = self.backbone.get_layer('conv5_block3_out').output

        x = self.aspp(encode4)

        # Decoder
        x = tf.image.resize(x, encode3.shape[1:3], method='bilinear')
        encode3 = self.compress_low_level_feature3(encode3)
        x = self.decode3(tf.concat([x, encode3], axis=-1))

        x = tf.image.resize(x, encode2.shape[1:3], method='bilinear')
        encode2 = self.compress_low_level_feature2(encode2)
        x = self.decode2(tf.concat([x, encode2], axis=-1))

        x = tf.image.resize(x, encode1.shape[1:3], method='bilinear')
        encode1 = self.compress_low_level_feature1(encode1)
        x = self.decode1(tf.concat([x, encode1], axis=-1))

        # Classifier
        x = self.classifier(x)
        self.outputs = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)

        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='EAR-Net')
        return model
