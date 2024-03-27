

#####################################################################################################
#                                                                                                    #
# ---------------------------------            FCN-8s             ---------------------------------#
#                                                                                                    #
######################################################################################################
# https://www.kaggle.com/code/abhinavsp0730/semantic-segmentation-by-implementing-fcn/notebook
# https://medium.com/geekculture/what-is-a-fcn-3135608d4903
# https://github.com/fmahoudeau/FCN-Segmentation-TensorFlow/blob/master/fcn_model.py
# https://github.com/sunnynevarekar/FCN/blob/master/src/models/fcn.py
# https://github.com/MarvinTeichmann/tensorflow-fcn


def FCN8s_model(num_classes=21, input_shape=(512,512,3)):

  inputs = tf.keras.Input(shape=input_shape)

  # Building a pre-trained VGG-16 feature extractor (i.e., without the final FC layers)
  backbone = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_tensor=inputs)

  # extract pool3, pool4, pool5 output tensors from backbone network
  pool3 = backbone.get_layer(name='block3_pool').output     # (64,64,256)
  pool4 = backbone.get_layer(name='block4_pool').output     # (32,32,512)
  pool5 = backbone.get_layer(name='block5_pool').output     # (16,16,512)

  # Replacing VGG dense layers by convolutions:
  f5_conv1 = tf.keras.layers.Conv2D(filters=4096, kernel_size=7, padding='same', activation='relu', name="conv6")(pool5) # 4096 in the original paper
  f5_conv2 = tf.keras.layers.Conv2D(filters=4096, kernel_size=1, padding='same', activation='relu', name="conv7")(f5_conv1) # 4096 in the original paper

  # FCN-32
  f5_pred = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, padding='same', activation='relu', name="conv_pred_1")(f5_conv2)      # (16,16,NB_CLASS)

  # Upsampling x2 FCN-16
  f5_upsampling = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name="upsampling1")(f5_pred)
  f6_conv1 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, padding='same', activation='relu', name="conv_pred_2")(pool4)
  f6_merge = tf.keras.layers.Add()([f6_conv1, f5_upsampling])       # (32,32,NB_CLASS)

  # Upsampling x2 FCN-8
  f6_upsampling = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear", name="merge1_2x_pred")(f6_merge)
  f7_conv1 = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, padding='same', activation='relu', name="conv_pred_3")(pool3)
  f7_merge = tf.keras.layers.Add()([f7_conv1, f6_upsampling])       # (64,64,NB_CLASS)

  outputs = tf.keras.layers.Conv2D(num_classes, kernel_size=1, padding='same', kernel_initializer='he_normal')(f7_merge)
  outputs = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear", name="8x_pred")(outputs)
  fcn_model = tf.keras.models.Model(inputs, outputs)
  return fcn_model
