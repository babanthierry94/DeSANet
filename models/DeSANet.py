
class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, in_dim):
        super(SelfAttentionLayer, self).__init__()
        height, width, C = in_dim
        self.N = height * width
        # Learnable parameter for adjustment
        self.beta = tf.Variable(initial_value=tf.zeros(1), trainable=True, name="beta")
        # Learnable parameters for linear projections
        self.Wq = self.add_weight(shape=(C, C), initializer='glorot_uniform', trainable=True, name="Wq")
        self.Wk = self.add_weight(shape=(C, C), initializer='glorot_uniform', trainable=True, name="Wk")
        self.Wv = self.add_weight(shape=(C, C), initializer='glorot_uniform', trainable=True, name="Wv")

        self.reshape_first = tf.keras.layers.Reshape((height * width, C), name="reshape_first")
        self.reshape_last = tf.keras.layers.Reshape((height, width, C), name="reshape_last")

        # self.relu = tf.keras.layers.ReLU()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        # flattened_x = tf.reshape(x, (batch_size, height * width, C))
        flattened_x = self.reshape_first(x)
        # Linear projections
        proj_query = tf.matmul(flattened_x, self.Wq)
        proj_key = tf.matmul(flattened_x, self.Wk)
        proj_value = tf.matmul(flattened_x, self.Wv)
        # Transpose to perform dot product efficiently
        proj_query = tf.transpose(proj_query, perm=[0, 2, 1])
        # Energy calculation (dot product between query and key)
        energy = tf.matmul(proj_key, proj_query)
        # Softmax normalization to obtain attention weights
        attention = tf.nn.softmax(energy / math.sqrt(self.N))
        # Output calculation based on attention weights
        scores = tf.matmul(attention, proj_value)
        # Reshape and adjust output with the beta parameter
        y = self.reshape_last(scores)
        # out = self.beta * out + x
        y = self.sigmoid(y)
        out = y * x
        return out


class DeSANet(object):

    def __init__(self, num_classes=21, backbone="resnet101", input_shape=(512,512,3), finetune=True):
        if backbone not in ['resnet101', 'resnet50']:
            print("backbone_name ERROR! Please input: resnet101, resnet50")
            raise NotImplementedError

        if finetune :
            self.pretrained = "imagenet"
        else :
            self.pretrained = None

        self.inputs = tf.keras.layers.Input(shape=input_shape)
        self.backbone_name = backbone
        self.num_classes = num_classes
        in_dim = (32, 32, 1024)
        self.self_att = SelfAttentionLayer(in_dim=(32, 32, 1024))

        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='mid_DeepLabv3p_v2')
        return model

    def build_network(self):
        low_level_feat, middle_level_feat, high_level_feat = self.build_encoder()
        self.outputs = self.build_decoder(low_level_feat, middle_level_feat, high_level_feat)

    def build_encoder(self):
        print("-------Backbone : %s-----" % self.backbone_name)
        if self.backbone_name == 'resnet50':
            backbone_model = tf.keras.applications.ResNet50(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            first_layer_name = "conv2_block3_out"
            middle_layer_name = "conv3_block4_out"
            last_layer_name = "conv4_block6_out"

        elif self.backbone_name == 'resnet101':
            backbone_model = tf.keras.applications.ResNet101(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            first_layer_name = "conv2_block3_out"
            middle_layer_name = "conv3_block4_out"
            last_layer_name = "conv4_block23_out"

        low_level_feat = backbone_model.get_layer(first_layer_name).output
        middle_level_feat = backbone_model.get_layer(middle_layer_name).output
        high_level_feat = backbone_model.get_layer(last_layer_name).output
      
        return low_level_feat, middle_level_feat, high_level_feat

  
    def build_decoder(self, low_level_feat, middle_level_feat, high_level_feat):

        high_level_feat = self.self_att(high_level_feat)

        high_features = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal')(high_level_feat)
        high_features = tf.keras.layers.BatchNormalization()(high_features)
        high_features = tf.keras.layers.Activation('relu')(high_features)
        high_features = tf.keras.layers.UpSampling2D(name="Decoder_Upsampling1a", size=(4,4), interpolation="bilinear")(high_features) #Upsampling x4

        middle_features = tf.keras.layers.Conv2D(128, kernel_size=1, padding='same', kernel_initializer='he_normal')(middle_level_feat)
        middle_features = tf.keras.layers.BatchNormalization()(middle_features)
        middle_features = tf.keras.layers.Activation('relu')(middle_features)
        middle_features =  tf.keras.layers.UpSampling2D(name="Decoder_Upsampling1b", size=(2,2), interpolation="bilinear")(middle_features) #Upsampling x2

        low_features = tf.keras.layers.Conv2D(64, kernel_size=1, padding='same', kernel_initializer='he_normal')(low_level_feat)
        low_features = tf.keras.layers.BatchNormalization()(low_features)
        low_features = tf.keras.layers.Activation('relu')(low_features)

        x = tf.keras.layers.Concatenate()([high_features, middle_features, low_features])

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(self.num_classes, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.UpSampling2D(size=(4,4), interpolation="bilinear")(x)
        outputs = x
        return outputs
