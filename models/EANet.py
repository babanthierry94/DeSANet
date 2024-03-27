
######################################################################################################
#                                                                                                    #
# --------------------------------------------- EANet -----------------------------------------------#
#                                                                                                    #
######################################################################################################

class EANet():
    def __init__(self, nclass, backbone='resnet50', input_shape=(512, 512, 3)):
        self.input_shape = input_shape
        self.pretrained = "imagenet"
        self.inputs = tf.keras.layers.Input(shape=self.input_shape)
        self.backbone = backbone

        height, width, in_channels = (32, 32, 1024)
        # resnet101v2
        if self.backbone == 'resnet101':
            # height, width, in_channels = (16, 16, 2048)
            height, width, in_channels = (32, 32, 1024)
        # resnet50v2
        elif self.backbone == 'resnet50':
            # height, width, in_channels = (16, 16, 2048)
            height, width, in_channels = (32, 32, 1024)
        # # efficientNetB4EfficientNetB4
        # elif self.backbone == 'efficientNetB4':
        #     # height, width, in_channels = (16, 16, 2048)
        #     height, width, in_channels = (32, 32, 960)
        # elif self.backbone == 'xception':
        #     # height, width, in_channels = (16, 16, 2048)
        #     height, width, in_channels = (32, 32, 1024)


        self.fc0 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.1)
        ])

        self.ext_head = ExternalAttention(height, width, 512)

        self.fc1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.1)
        ])

        self.fc2 = tf.keras.layers.Conv2D(nclass, kernel_size=1, padding="same")
        self.upsampling1 = tf.keras.layers.UpSampling2D(size=self.inputs.shape[1]//height, interpolation="bilinear")

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
        # # efficientNetB4
        # elif self.backbone == 'efficientNetB4':
        #     backbone_model = tf.keras.applications.EfficientNetB4(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
        #     x = backbone_model.get_layer(name="block6a_expand_activation").output # 32, 32, 960
        # # xception
        # elif self.backbone == 'xception':
        #     backbone_model = tf.keras.applications.Xception(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
        #     x = backbone_model.get_layer(name="block13_sepconv2_bn").output # 32, 32, 1024
        #     # x = self.relu(x)

        # x = backbone_model.get_layer(index=-1).output
        #block4
        block4 = self._bottleneck_resblock(third_block_layer, 2048, 1, 2, 'conv5_block1', identity_connection=False)
        for i in range(2, 4):
            block4 = self._bottleneck_resblock(block4, 2048, 1, 1, 'conv5_block%d'%i)

        x = self.fc0(block4)
        x = self.ext_head(x)
        x = self.fc1(x)
        x = self.fc2(x)
        outputs = self.upsampling1(x)

        # Create the DANet model
        model = tf.keras.Model(inputs=self.inputs, outputs=outputs, name='EANet')
        return model
