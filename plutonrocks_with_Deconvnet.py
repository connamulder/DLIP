"""
    @Project: projectTransfer
    @File   : plutonrocks_with_Decovnet.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2020-10-11
    @Modify :
    @info   :
"""

import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, InputLayer
from keras.layers import Flatten, Activation, Conv2D, Dense
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling2D
from keras_preprocessing import image

from keras.models import Model
import keras.backend as K

import math
from PIL import Image
import six
import imageio
from PIL.JpegImagePlugin import JpegImageFile
import os

os.environ["CUDA_VISIBLE_DEVICE"] = "1"


class DConvolution2D(object):

    def __init__(self, layer):
        self.layer = layer

        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        filters = W.shape[3]
        up_row = W.shape[0]
        up_col = W.shape[1]
        input_img = keras.layers.Input(shape=layer.input_shape[1:])

        output = keras.layers.Conv2D(filters, (up_row, up_col), kernel_initializer=tf.constant_initializer(W),
                                     bias_initializer=tf.constant_initializer(b), padding='same')(input_img)
        self.up_func = K.function([input_img, K.learning_phase()], [output])
        # Deconv filter (exchange no of filters and depth of each filter)
        W = np.transpose(W, (0, 1, 3, 2))
        # Reverse columns and rows
        W = W[::-1, ::-1, :, :]
        down_filters = W.shape[3]
        down_row = W.shape[0]
        down_col = W.shape[1]
        b = np.zeros(down_filters)
        input_d = keras.layers.Input(shape=layer.output_shape[1:])

        output = keras.layers.Conv2D(down_filters, (down_row, down_col), kernel_initializer=tf.constant_initializer(W),
                                     bias_initializer=tf.constant_initializer(b), padding='same')(input_d)
        self.down_func = K.function([input_d, K.learning_phase()], [output])

    def up(self, data, learning_phase=0):
        # Forward pass
        self.up_data = self.up_func([data, learning_phase])
        self.up_data = np.squeeze(self.up_data, axis=0)
        self.up_data = numpy.expand_dims(self.up_data, axis=0)
        # print(self.up_data.shape)
        return self.up_data

    def down(self, data, learning_phase=0):
        # Backward pass
        self.down_data = self.down_func([data, learning_phase])
        self.down_data = np.squeeze(self.down_data, axis=0)
        self.down_data = numpy.expand_dims(self.down_data, axis=0)
        # print(self.down_data.shape)
        return self.down_data


class DActivation(object):
    def __init__(self, layer, linear=False):
        self.layer = layer
        self.linear = linear
        self.activation = layer.activation
        input = K.placeholder(shape=layer.output_shape)

        output = self.activation(input)
        # According to the original paper,
        # In forward pass and backward pass, do the same activation(relu)
        # Up method
        self.up_func = K.function(
            [input, K.learning_phase()], [output])
        # Down method
        self.down_func = K.function(
            [input, K.learning_phase()], [output])

    def up(self, data, learning_phase=0):
        self.up_data = self.up_func([data, learning_phase])
        self.up_data = np.squeeze(self.up_data, axis=0)
        self.up_data = numpy.expand_dims(self.up_data, axis=0)
        print(self.up_data.shape)
        return self.up_data

    def down(self, data, learning_phase=0):
        self.down_data = self.down_func([data, learning_phase])
        self.down_data = np.squeeze(self.down_data, axis=0)
        self.down_data = numpy.expand_dims(self.down_data, axis=0)
        print(self.down_data.shape)
        return self.down_data


class DInput(object):

    def __init__(self, layer):
        self.layer = layer

    # input and output of Inputl layer are the same
    def up(self, data, learning_phase=0):
        self.up_data = data
        return self.up_data

    def down(self, data, learning_phase=0):
        # data=np.squeeze(data,axis=0)
        data = numpy.expand_dims(data, axis=0)
        self.down_data = data
        return self.down_data


class DDense(object):

    def __init__(self, layer):
        self.layer = layer
        weights = layer.get_weights()
        W = weights[0]
        b = weights[1]

        # Up method
        input = Input(shape=layer.input_shape[1:])
        output = keras.layers.Dense(layer.output_shape[1],
                                    kernel_initializer=tf.constant_initializer(W),
                                    bias_initializer=tf.constant_initializer(b))(input)
        self.up_func = K.function([input, K.learning_phase()], [output])

        # Transpose W  for down method
        W = W.transpose()
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape
        b = np.zeros(self.input_shape[1])
        flipped_weights = [W, b]
        input = Input(shape=self.output_shape[1:])
        output = keras.layers.Dense(self.input_shape[1:],
                                    kernel_initializer=tf.constant_initializer(W),
                                    bias_initializer=tf.constant_initializer(b))(input)
        self.down_func = K.function([input, K.learning_phase()], [output])

    def up(self, data, learning_phase=0):
        self.up_data = self.up_func([data, learning_phase])
        self.up_data = np.squeeze(self.up_data, axis=0)
        self.up_data = numpy.expand_dims(self.up_data, axis=0)
        print(self.up_data.shape)
        return self.up_data

    def down(self, data, learning_phase=0):
        self.down_data = self.down_func([data, learning_phase])
        self.down_data = np.squeeze(self.down_data, axis=0)
        self.down_data = numpy.expand_dims(self.down_data, axis=0)
        print(self.down_data.shape)
        return self.down_data


class DFlatten(object):
    def __init__(self, layer):
        self.layer = layer
        self.shape = layer.input_shape[1:]
        self.up_func = K.function(
            [layer.input, K.learning_phase()], [layer.output])

    # Flatten 2D input into 1D output
    def up(self, data, learning_phase=0):
        self.up_data = self.up_func([data, learning_phase])
        self.up_data = np.squeeze(self.up_data, axis=0)
        self.up_data = numpy.expand_dims(self.up_data, axis=0)
        print(self.up_data.shape)
        return self.up_data

    # Reshape 1D input into 2D output
    def down(self, data, learning_phase=0):
        new_shape = [data.shape[0]] + list(self.shape)
        assert np.prod(self.shape) == np.prod(data.shape[1:])
        self.down_data = np.reshape(data, new_shape)
        return self.down_data


class DPooling(object):

    def __init__(self, layer):

        self.layer = layer
        self.poolsize = layer.pool_size

    def up(self, data, learning_phase=0):

        [self.up_data, self.switch] = \
            self.__max_pooling_with_switch(data, self.poolsize)
        return self.up_data

    def down(self, data, learning_phase=0):

        self.down_data = self.__max_unpooling_with_switch(data, self.switch)
        return self.down_data

    def __max_pooling_with_switch(self, input, poolsize):

        switch = np.zeros(input.shape)
        out_shape = list(input.shape)
        row_poolsize = int(poolsize[0])
        col_poolsize = int(poolsize[1])
        print('row_poolsize:' + str(row_poolsize))
        print('row_poolsize:' + str(col_poolsize))
        out_shape[1] = math.floor(out_shape[1] / poolsize[0])
        out_shape[2] = math.floor(out_shape[2] / poolsize[1])
        print(out_shape)
        pooled = np.zeros(out_shape)

        for sample in range(input.shape[0]):
            for dim in range(input.shape[3]):
                for row in range(out_shape[1]):
                    for col in range(out_shape[2]):
                        patch = input[sample,
                                row * row_poolsize: (row + 1) * row_poolsize,
                                col * col_poolsize: (col + 1) * col_poolsize, dim]
                        max_value = patch.max()
                        pooled[sample, row, col, dim] = max_value
                        max_col_index = patch.argmax(axis=1)
                        max_cols = patch.max(axis=1)
                        max_row = max_cols.argmax()
                        max_col = max_col_index[max_row]
                        switch[sample,
                               row * row_poolsize + max_row,
                               col * col_poolsize + max_col,
                               dim] = 1
        return [pooled, switch]

    # Compute unpooled output using pooled data and switch
    def __max_unpooling_with_switch(self, input, switch):


        print('switch1 ' + str(switch.shape))
        print('input  ' + str(input.shape))
        tile = np.ones((math.floor(switch.shape[1] / input.shape[1]),
                        math.floor(switch.shape[2] / input.shape[2])))
        print('tile ' + str(tile.shape))
        tile = numpy.expand_dims(tile, axis=3)
        input = numpy.squeeze(input, axis=0)
        out = np.kron(input, tile)
        print('out ' + str(out.shape))
        #switch = numpy.squeeze(switch, axis=0)
        print('switch2 ' + str(switch.shape))
        # unpooled = out * switch
        if out.shape[1] == switch.shape[2]:
            unpooled = out * switch
            unpooled = numpy.expand_dims(unpooled, axis=0)
        else:
            outlist = []
            for i in range(out.shape[2]):
                outtemp = out[:, :, i]
                outtempZeroPadding = np.pad(outtemp, (1, 1), 'constant', constant_values=(0, 0))
                outlist.append(outtempZeroPadding)
            outstack = np.dstack(outlist)
            unpooled = outstack * switch
            # unpooled = np.dot(out, switch)
            unpooled = numpy.expand_dims(unpooled, axis=0)
        return unpooled


def visualize(model, data, layer_name, feature_to_visualize, visualize_mode):
    deconv_layers = []
    # Stack layers
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], Conv2D):
            deconv_layers.append(DConvolution2D(model.layers[i]))
            deconv_layers.append(
                DActivation(model.layers[i]))
        elif isinstance(model.layers[i], MaxPooling2D):
            deconv_layers.append(DPooling(model.layers[i]))
        elif isinstance(model.layers[i], Dense):
            deconv_layers.append(DDense(model.layers[i]))
            deconv_layers.append(
                DActivation(model.layers[i]))
        elif isinstance(model.layers[i], Activation):
            deconv_layers.append(DActivation(model.alyers[i]))
        elif isinstance(model.layers[i], Flatten):
            deconv_layers.append(DFlatten(model.layers[i]))
        elif isinstance(model.layers[i], InputLayer):
            deconv_layers.append(DInput(model.layers[i]))
        elif isinstance(model.layers[i], ZeroPadding2D):
            print(model.layers[i])
        else:
            print('Cannot handle this type of layer')
            print(model.layers[i].get_config())
            sys.exit()
        if layer_name == model.layers[i].name:
            break

    # Forward pass
    deconv_layers[0].up(data)
    for i in range(1, len(deconv_layers)):
        print("UP layer" + str(i))
        deconv_layers[i].up(deconv_layers[i - 1].up_data)

    output = deconv_layers[-1].up_data
    print(output.shape)
    assert output.ndim == 2 or output.ndim == 4
    if output.ndim == 2:
        feature_map = output[:, feature_to_visualize]
    else:
        feature_map = output[:, :, :, feature_to_visualize]
    if 'max' == visualize_mode:
        max_activation = feature_map.max()
        temp = feature_map == max_activation
        feature_map = feature_map * temp
    elif 'all' != visualize_mode:
        print('Illegal visualize mode')
        sys.exit()
    output = np.zeros_like(output)
    if 2 == output.ndim:
        output[:, feature_to_visualize] = feature_map
    else:
        output[:, :, :, feature_to_visualize] = feature_map

    # Backward pass
    deconv_layers[-1].down(output)
    for i in range(len(deconv_layers) - 2, -1, -1):
        print("DOWN LAYER:" + str(i))
        deconv_layers[i].down(deconv_layers[i + 1].down_data)
    deconv = deconv_layers[0].down_data
    deconv = deconv.squeeze()

    return deconv


# Define the network architecture of AlexNet model
def pre_trained_AlexNet(input_shape, num_classes):
    X_input = Input(input_shape)

    X = Conv2D(64, (11, 11), strides=(4, 4), padding='same', activation='relu')(X_input)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = Conv2D(192, (5, 5), padding='same', activation='relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = Conv2D(384, (3, 3), padding='same', activation='relu')(X)
    X = Conv2D(256, (3, 3), padding='same', activation='relu')(X)
    X = Conv2D(256, (3, 3), padding='same', activation='relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = Flatten(name='flatten')(X)

    X = Dense(1024, activation='relu', name='fc1024')(X)
    X = Dropout(0.5)(X)
    X = Dense(num_classes, activation='softmax')(X)
    model_Alex_pluton = Model(inputs=X_input, outputs=X, name='alexnet_pluton')

    weight_path = "E:\\01_Paper_Temp\\2020_正负片麻岩特征提取\\AlexNet_TL_89.9_20200929_Par=1192_bs=32\\saved_tl_models_log\\plutonrocks_augu_227_pre_trained_AlexNet_model.014.h5"
    model_Alex_pluton.load_weights(weight_path)

    return model_Alex_pluton


def resize_image(in_image: Image, new_width, new_height, crop_or_pad=True):
    """ Resize an image.
    Arguments:
        in_image: `PIL.Image`. The image to resize.
        new_width: `int`. The image new width.
        new_height: `int`. The image new height.
        crop_or_pad: Whether to resize as per tensorflow's function
    Returns:
        `PIL.Image`. The resize image.
    """
    # img = in_image.copy()
    img = in_image

    if crop_or_pad:
        if isinstance(img, six.string_types):
            tempimg = Image.open(img)
            half_width = tempimg.size[0] // 2
            half_height = tempimg.size[1] // 2
        elif isinstance(img, imageio.core.util.Image):
            # img = Image.fromarray(incoming)
            tempimg = Image.fromarray(img)
            half_width = tempimg.width // 2
            half_height = tempimg.height // 2
        elif isinstance(img, JpegImageFile):
            half_width = img.width // 2
            half_height = img.height // 2
        elif len(img.shape) >= 2:
            half_width = img.shape[0] // 2
            half_height = img.shape[1] // 2

        half_new_width = new_width // 2
        half_new_height = new_height // 2

        img = img.crop((half_width - half_new_width,
                        half_height - half_new_height,
                        half_width + half_new_width,
                        half_height + half_new_height
                        ))

    img = img.resize(size=(new_width, new_height))
    return img


def load_image(path, preprocess=True, imagesize=227):
    im = image.load_img(path)

    if im.size[0] > im.size[1]:
        height = im.size[1]
    else:
        height = im.size[0]

    im = resize_image(im, height, height, crop_or_pad=True)
    img_temp = resize_image(im, imagesize, imagesize, False)

    if preprocess:
        img_temp = image.img_to_array(img_temp)
        img_temp = (img_temp / 255.0)
        img_temp = np.expand_dims(img_temp, axis=0)
    return img_temp


def main():
    # set the input image file path
    img_path = "E:\\LIME_pluton-rocks_224\\62_monzonitic_granite\\P1280585.jpg"

    # set the name of the layer which feature maps will be shown
    layer_name = 'conv2d_5'
    # the top 9 activations in the feature maps of the layer should be pre-calculate.
    feature_to_visualize_list = [253, 161, 157, 162, 135, 57, 104, 200, 54]
    # set a default number  of the activations to visualize, it will be changed to the number in the feature_to_visualize_list
    feature_to_visualize = 56
    visualize_mode_list = ['all']

    # type of the model
    model_type = 'pre_trained_AlexNet'

    # parameters of the plutonic images dataset
    if model_type == 'pre_trained_Xception':
        img_width = 299
        img_height = 299
    elif model_type == 'pre_trained_AlexNet':
        img_width = 227
        img_height = 227
    else:
        img_width = 224
        img_height = 224
    input_shape = (img_width, img_height, 3)
    num_classes = 10

    if model_type == 'pre_trained_AlexNet':
        model = pre_trained_AlexNet(input_shape, num_classes)

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    print(model.summary())
    if not layer_dict.__contains__(layer_name):
        print('Wrong layer name')
        sys.exit()

    # Load data and preprocess
    img_array = load_image(img_path, preprocess=True, imagesize=img_width)

    for feature_to_visualize in feature_to_visualize_list:
        for visualize_mode in visualize_mode_list:
            deconv = visualize(model, img_array,
                               layer_name, feature_to_visualize, visualize_mode)
            print(deconv.shape)
            deconv = deconv - deconv.min()
            deconv *= 1.0 / (deconv.max() + 1e-8)
            uint8_deconv = (deconv * 255).astype(np.uint8)
            img = Image.fromarray(uint8_deconv, 'RGB')
            img.save('{}_{}_kernel{}_{}.png'.format(img_path[0:-4], layer_name, feature_to_visualize, visualize_mode))


if "__main__" == __name__:
    main()