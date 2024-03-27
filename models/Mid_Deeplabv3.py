import tensorflow as tf
from Attention_modules import SimAM

######################################################################################################
#                                                                                                    #
# ----------------------------------------- Mid-DeepLabv3+ ------------------------------------------#
#                                                                                                    #
######################################################################################################

class mid_DeepLabv3p(object):
    """
    ResNet-101
    ResNet-50
    """
    def __init__(self, num_classes=21, backbone="resnet50", input_shape=(512,512,3), finetune=True):
        if backbone not in ['resnet50', 'resnet101']:
            print("backbone_name ERROR! Please input: resnet50, resnet101")
            raise NotImplementedError

        self.inputs = tf.keras.layers.Input(shape=input_shape)
        # Number of blocks for ResNet50 and ResNet101
        self.backbone_name = backbone
        self.num_classes = num_classes

        if finetune :
            self.pretrained = "imagenet"
        else :
            self.pretrained = None

        self.Attention1 = SimAM()
        self.Attention2 = SimAM()
        self.Attention3 = SimAM()
        # Dilations rates for ASPP module
        self.aspp_dilations = [1, 6, 12, 18]
        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='mid_DeepLabv3p')
        return model

    def build_network(self):
        low_level_feat, middle_level_feat, high_level_feat = self.build_encoder()
        self.outputs = self.build_decoder(low_level_feat, middle_level_feat, high_level_feat)

    def build_encoder(self):

        if self.backbone_name == 'resnet50':
            backbone_model = tf.keras.applications.ResNet50(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            first_layer_name = "conv2_block3_2_relu"
            middle_layer_name = "conv3_block4_2_relu"
            last_layer_name = "conv4_block6_2_relu"

        elif self.backbone_name == 'resnet101':
            backbone_model = tf.keras.applications.ResNet101(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            first_layer_name = "conv2_block3_2_relu"
            middle_layer_name = "conv3_block4_2_relu"
            last_layer_name = "conv4_block23_2_relu"

        low_level_feat = backbone_model.get_layer(first_layer_name).output
        middle_level_feat = backbone_model.get_layer(middle_layer_name).output
        high_level_feat = backbone_model.get_layer(last_layer_name).output

        return low_level_feat, middle_level_feat, high_level_feat


    def build_decoder(self, low_level_feat, middle_level_feat, high_level_feat):

        high_level_feat = self.Attention1(high_level_feat)
        middle_level_feat = self.Attention2(middle_level_feat)
        low_level_feat = self.Attention3(low_level_feat)

        # ASPP module
        high_features = self._ASPPv2(high_level_feat, 256, self.aspp_dilations)

        high_features = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal')(high_features)
        high_features = tf.keras.layers.BatchNormalization()(high_features)
        high_features = tf.keras.layers.Activation('relu')(high_features)
        high_features = tf.keras.layers.UpSampling2D(size=(4,4), interpolation="bilinear")(high_features) #Upsampling x4

        middle_features = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', kernel_initializer='he_normal')(middle_level_feat)
        middle_features = tf.keras.layers.BatchNormalization()(middle_features)
        middle_features = tf.keras.layers.ReLU()(middle_features)
        middle_features =  tf.keras.layers.UpSampling2D(name="Decoder_Upsampling1b", size=(2,2), interpolation="bilinear")(middle_features) #Upsampling x2

        low_features = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', kernel_initializer='he_normal')(low_level_feat)
        low_features = tf.keras.layers.BatchNormalization()(low_features)
        low_features = tf.keras.layers.Activation('relu')(low_features)

        x = tf.keras.layers.Concatenate()([high_features, middle_features, low_features])

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.UpSampling2D(size=self.inputs.shape[1]//x.shape[1], interpolation="bilinear")(x)
        outputs = tf.keras.layers.Conv2D(self.num_classes, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

        return outputs

    def _Atrous_SepConv(self, x, conv_type="sepconv2d", prefix="None", filters=256, kernel_size=3,  stride=1, dilation_rate=1, use_bias=False):
        conv_dict = {
            'conv2d': tf.keras.layers.Conv2D,
            'sepconv2d': tf.keras.layers.SeparableConv2D
        }
        conv = conv_dict[conv_type]
        x = conv(filters, kernel_size, name=prefix, strides=stride, dilation_rate=dilation_rate, padding="same", kernel_initializer='he_normal', use_bias=use_bias)(x)
        x = tf.keras.layers.BatchNormalization(name='%s_bn'%prefix)(x)
        x = tf.keras.layers.Activation('relu', name='%s_relu'%prefix)(x)
        return x

    def _ASPPv2(self, x, nb_filters, d):
        x1 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv1', filters=nb_filters, kernel_size=1, dilation_rate=d[0], use_bias=True)
        x2 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv2', filters=nb_filters, kernel_size=3, dilation_rate=d[1], use_bias=True)
        x3 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv3', filters=nb_filters, kernel_size=3, dilation_rate=d[2], use_bias=True)
        x4 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv4', filters=nb_filters, kernel_size=3, dilation_rate=d[3], use_bias=True)

        x5 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='aspp/avg')(x)
        x5 = tf.keras.layers.Conv2D(256, kernel_size=1)(x5)
        x5 = tf.keras.layers.UpSampling2D(size=x.shape[1] // x5.shape[1], interpolation="bilinear", name='asspv2/avg_upsambling')(x5)
        out = tf.keras.layers.Concatenate(name='aspp/add')([x1, x2, x3, x4, x5])
        return out
