"""
    @Project: projectTransfer
    @File   : plutonrocks_with_LIME.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2020-09-17
    @Modify :
    @info   :
"""

import os
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Flatten, Convolution2D, GlobalAveragePooling2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception

from keras_preprocessing import image
from PIL import Image
import six
import imageio
from PIL.JpegImagePlugin import JpegImageFile
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries


def pre_trained_xception(input_shape, num_classes):
    X_input = Input(input_shape)

    Xception_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape
                          )

    top_model = Xception_model(X_input)
    top_model = GlobalAveragePooling2D(name='avg_pool')(top_model)
    predictions = Dense(num_classes, activation='softmax', name='fc10')(top_model)
    Xception_model_pluton = Model(inputs=X_input, outputs=predictions)

    weight_path = "E:\\01_Paper_Temp\\2020_正负片麻岩特征提取\\Xception_TL_27.2_20200923_Par=2083_bs=32\\saved_tl_models_log\\plutonrocks_augu_299_pre_trained_Xception_model.006.h5"
    Xception_model_pluton.load_weights(weight_path)

    return Xception_model_pluton


def pre_trained_resnet50(input_shape, num_classes):
    X_input = Input(input_shape)

    resnet50_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape
                              )

    top_model = resnet50_model(X_input)
    top_model = GlobalAveragePooling2D(name='avg_pool')(top_model)
    predictions = Dense(num_classes, activation='softmax', name='fc10')(top_model)
    resnet50_model_pluton = Model(inputs=X_input, outputs=predictions)

    weight_path = "E:\\01_Paper_Temp\\2020_正负片麻岩特征提取\\ResNet50_TL_83.6_20200923_Par=2355_bs=32\\saved_tl_models_log\\plutonrocks_augu_224_pre_trained_ResNet50_model.006.h5"
    resnet50_model_pluton.load_weights(weight_path)

    return resnet50_model_pluton


def pre_trained_vgg16(input_shape, num_classes):
    model_vgg = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=input_shape
                       )

    model = Flatten(name='flatten')(model_vgg.output)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dense(num_classes, activation='softmax')(model)
    model_vgg_pluton = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pluton')

    weight_path = "E:\\01_Paper_Temp\\2020_正负片麻岩特征提取\\VGG16_TL_89.4_20200922_Param=13410_batchsize=32\\saved_tl_models_log\\plutonrocks_augu_224_pre_trained_VGG16_model.022.h5"
    model_vgg_pluton.load_weights(weight_path)

    return model_vgg_pluton


def pre_trained_AlexNet(input_shape, num_classes):
    X_input = Input(input_shape)

    X = ZeroPadding2D((2, 2), input_shape=(227, 227, 3))(X_input)
    X = Convolution2D(64, (11, 11), strides=(4, 4), activation='relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = ZeroPadding2D((2, 2))(X)
    X = Convolution2D(192, (5, 5), activation='relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Convolution2D(384, (3, 3), activation='relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Convolution2D(256, (3, 3), activation='relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Convolution2D(256, (3, 3), activation='relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    model = Flatten(name='flatten')(X)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dropout(0.5)(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(0.5)(model)
    model = Dense(num_classes, activation='softmax')(model)
    model_Alex_pluton = Model(inputs=X_input, outputs=model, name='alexnet_pluton')

    weight_path = "E:\\01_Paper_Temp\\2020_正负片麻岩特征提取\\AlexNet_TL_89.7_20200921_Param=5704_batchsize=32\\saved_tl_models_log\\plutonrocks_augu_227_pre_trained_AlexNet_model.028.h5"
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
    img = in_image

    if crop_or_pad:
        if isinstance(img, six.string_types):
            tempimg = Image.open(img)
            half_width = tempimg.size[0] // 2
            half_height = tempimg.size[1] // 2
        elif isinstance(img, imageio.core.util.Image):
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


def load_image(path, preprocess=True):
    # read a input image to predict
    im = image.load_img(path)

    if im.size[0] > im.size[1]:
        height = im.size[1]
    else:
        height = im.size[0]

    im = resize_image(im, height, height, crop_or_pad=True)
    img_temp = resize_image(im, img_width, img_width, False)

    if preprocess:
        img_temp = image.img_to_array(img_temp)
        img_temp = (img_temp / 255.0)
        img_temp = np.expand_dims(img_temp, axis=0)
    return img_temp


def decode_predictions_custom(preds, top=5):
    CLASS_CUSTOM = ["Pyroxene_hornblendite", "Gabbro", "Diorite", "Quartz_diorite", "Quartz_monzodiorite",
                 "Quartz_monzonite", "Syenogranite", "Monzonitic_granite", "Granodiorite", "Nepheline_syenite"]
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_CUSTOM[i]) + (pred[i]*100,) for i in top_indices]
        results.append(result)
    return results


if __name__ == "__main__":
    # type of the model, such as 'pre_trained_AlexNet', 'pre_trained_VGG16', 'pre_trained_ResNet50' and 'pre_trained_Xception'
    model_type = 'pre_trained_AlexNet'

    num_features = 1
    image_path_list = []

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

    if model_type == 'pre_trained_VGG16':
        model = pre_trained_vgg16(input_shape, num_classes)
    elif model_type == 'pre_trained_AlexNet':
        model = pre_trained_AlexNet(input_shape, num_classes)
    elif model_type == 'pre_trained_ResNet50':
        model = pre_trained_resnet50(input_shape, num_classes)
    elif model_type == 'pre_trained_Xception':
        model = pre_trained_xception(input_shape, num_classes)

    # append the file path of the input image to the list
    image_path_list.append("E:\\LIME_pluton-rocks\\71_nepheline_syenites\\P1290300.jpg")

    for image_path in image_path_list:
        filename = os.path.basename(image_path)
        filedirname = os.path.dirname(image_path)
        index = filename.find(".", 0)
        filename = filename[0:index]
        print("The file name is: ", filename)

        # read one image
        imx = load_image(image_path)
        # predict the rock type of the image
        im_pre = model.predict(imx)
        print(im_pre)
        top = np.argmax(im_pre)
        print("the rock type was predicted as：", top)
        p = decode_predictions_custom(im_pre, top=3)
        for x in p[0]:
            print(x)

        explainer = lime_image.LimeImageExplainer()
        x = imx[0].astype(np.double)  # lime要求numpy array
        explanation = explainer.explain_instance(x, model.predict, top_labels=10, hide_color=0, num_samples=1000)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=num_features,
                                                    hide_rest=True)
        plt.imshow(mark_boundaries(temp, mask))
        map_name = '%s_%s_%s_%s_%s_%s.png' % ('LIME', model_type, num_features, filename, p[0][0], 'labels[0]')
        map_dir = os.path.join(filedirname, map_name)
        plt.savefig(map_dir)

