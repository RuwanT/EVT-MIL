"""
Contain network prototypes
"""


def AlexNet_modified(input_shape=None, regularize_weight=0.0001):

    """
    Alexnet convolution layers with added batch-normalization and regularization

    :param input_shape:
    :param regularize_weight:
    :return:
    """
    from keras.layers import Conv2D, Input, MaxPooling2D, ZeroPadding2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import concatenate
    from keras.models import Model
    from keras.regularizers import l2


    img_input = Input(shape=input_shape)

    #Branch A (mimic the original alexnet)
    x = Conv2D(48, (11, 11), strides=(4,4), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(img_input)
    x = MaxPooling2D((3,3), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = ZeroPadding2D((2, 2))(x)

    x = Conv2D(128, (5, 5), strides=(1,1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = ZeroPadding2D((1, 1))(x)

    x = Conv2D(192, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = ZeroPadding2D((1, 1))(x)

    x = Conv2D(192, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = ZeroPadding2D((1, 1))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)

    # Branch B (mimic the original alexnet)
    y = Conv2D(48, (11, 11), strides=(4, 4), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(img_input)
    y = MaxPooling2D((3, 3), strides=(2, 2))(y)
    y = BatchNormalization(axis=-1)(y)
    y = ZeroPadding2D((2, 2))(y)

    y = Conv2D(128, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(y)
    y = MaxPooling2D((3, 3), strides=(2, 2))(y)
    y = BatchNormalization(axis=-1)(y)
    y = ZeroPadding2D((1, 1))(y)

    y = Conv2D(192, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(y)
    y = BatchNormalization(axis=-1)(y)
    y = ZeroPadding2D((1, 1))(y)

    y = Conv2D(192, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(y)
    y = BatchNormalization(axis=-1)(y)
    y = ZeroPadding2D((1, 1))(y)

    y = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(y)
    y = MaxPooling2D((3, 3), strides=(2, 2))(y)
    y = ZeroPadding2D((1, 1))(y)

    out = concatenate([x,y], axis=-1)

    inputs = img_input
    model = Model(inputs, out, name='alexnet')

    return model


def mnistNet_modified(input_shape=None, regularize_weight=0.0001):

    """


    :param input_shape:
    :param regularize_weight:
    :return:
    """
    from keras.layers import Conv2D, Input, MaxPooling2D, ZeroPadding2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import concatenate
    from keras.models import Model
    from keras.regularizers import l2

    img_input = Input(shape=input_shape)

    x = Conv2D(64, (5, 5), strides=(1,1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(img_input)
    x = MaxPooling2D((2,2), strides=(2, 2))(x)

    x = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=l2(regularize_weight))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    inputs = img_input
    model = Model(inputs, x, name='alexnet')

    model.summary()

    return model


def squeeze_net(input_shape=None, regularize_weight=0.0001):
    from keras.layers import Input, Convolution3D, Concatenate, MaxPooling3D, GlobalAveragePooling3D, Dense
    from keras.models import Model
    from keras.regularizers import l2

    img_input = Input(shape=input_shape)

    def fire_module(x, filters, name="firemodule"):
        squeeze_filter, expand_filter1, expand_filter2 = filters
        squeeze = Convolution3D(squeeze_filter, (1, 1, 1), activation='relu', padding='same',
                                name=name + "_squeeze1x1x1", kernel_regularizer=l2(regularize_weight))(x)
        expand1 = Convolution3D(expand_filter1, (1, 1, 1), activation='relu', padding='same',
                                name=name + "_expand1x1x1", kernel_regularizer=l2(regularize_weight))(squeeze)
        expand2 = Convolution3D(expand_filter2, (3, 3, 3), activation='relu', padding='same',
                                name=name + "_expand3x3x3", kernel_regularizer=l2(regularize_weight))(squeeze)
        x = Concatenate(axis=-1, name=name)([expand1, expand2])
        return x

    x = Convolution3D(64, kernel_size=(3, 5, 5), strides=(1, 1, 1), padding="same", activation="relu", name='conv1', kernel_regularizer=l2(regularize_weight))(
        img_input)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool1', padding="valid")(x)

    # x = fire_module(x, (16, 64, 64), name="fire2")
    x = fire_module(x, (16, 64, 64), name="fire3")
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool3', padding="valid")(x)

    x = fire_module(x, (32, 128, 128), name="fire4")
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 2, 2), name='maxpool5', padding="valid")(x)

    # x = fire_module(x, (48, 192, 192), name="fire6")
    x = fire_module(x, (64, 256, 256), name="fire8")

    model = Model(img_input, x, name="squeezenet")


    return model