"""
    @Project: projectTransfer
    @File   : plutonrocks_with_TransferLearning.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2020-07-23
    @info   :
    -All the comparative experiments were performed using the open-source deep learning framework Tensorflow-gpu 1.13.1.
    -The deep CNNs were build, complied and evaluated in Keras-gpu 2.3.1.
    -The network performance was evaluated using the accuracy method.
"""

import keras
print("Keras:{}".format(keras.__version__))

import numpy as np
import os
from keras.models import Input, Model
from keras.layers import Dense, Conv2D
from keras.layers import Dropout, BatchNormalization
from keras.layers import Flatten, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.applications import ResNet50, VGG16, Xception
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

import datetime
from math import pow, floor

DATASETSLIB_HOME = os.path.expanduser('~/datasetslib')
import sys
if not DATASETSLIB_HOME in sys.path:
    sys.path.append(DATASETSLIB_HOME)
import datasetslib

datasetslib.datasets_root = 'E:\\datasets'
models_root = os.path.join(os.path.expanduser('~'), 'models')

print("datasets_root:", datasetslib.datasets_root)
print("models_root:", models_root)


# type of the model, such as 'pre_trained_AlexNet', 'pre_trained_VGG16', 'pre_trained_ResNet50' and 'pre_trained_Xception'
model_type = 'pre_trained_AlexNet'

# parameters of the plutonic images dataset
if model_type == 'pre_trained_Xception':
    img_width = 299
    img_height = 299
    lr_ini = 1e-4
elif model_type == 'pre_trained_AlexNet':
    img_width = 227
    img_height = 227
    lr_ini = 1e-4
else:
    img_width = 224
    img_height = 224
    lr_ini = 1e-7
input_shape = (img_width, img_height, 3)
num_classes = 10

DATASET = 'simpleplutonrocks_amsr_%s' % img_width
print("dataset_name:", DATASET)

epochs = 1000
lrate = 0.01
decay = lrate/epochs
batch_size = 32

subtract_mean = True
seed = 7
np.random.seed(seed)

data_augmentation = False
flow_from_dir = True

# set the save paths of the finish training model and the dataset
save_dir = os.path.join(os.getcwd(), 'saved_tl_models_log')
dataset_dir = os.path.join(datasetslib.datasets_root, DATASET)

# set the file paths of the training dataset and the validation dataset
validate_dir = os.path.join(dataset_dir, "val")
train_dir = os.path.join(dataset_dir, "train")

model_name = DATASET+'_'+'%s_model.{epoch:03d}.h5' % model_type
h5log_file = os.path.join(save_dir, 'h5log.txt')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
print("Model Type: ", model_type)
print("Model File: ", filepath)


def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def setup_to_transfer_learning(base_model):
    for layer in base_model.layers:
        layer.trainable = False


def pre_trained_xception(input_shape, num_classes):
    K.set_learning_phase(0)
    X_input = Input(input_shape)

    Xception_model = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape
                          )

    K.set_learning_phase(1)

    top_model = Xception_model(X_input)
    top_model = GlobalAveragePooling2D(name='avg_pool')(top_model)
    predictions = Dense(num_classes, activation='softmax', name='fc10')(top_model)
    Xception_model_pluton = Model(inputs=X_input, outputs=predictions)

    return Xception_model_pluton


def pre_trained_resnet50(input_shape, num_classes):
    K.set_learning_phase(0)
    X_input = Input(input_shape)

    resnet50_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape
                              )

    K.set_learning_phase(1)

    top_model = resnet50_model(X_input)
    top_model = GlobalAveragePooling2D(name='avg_pool')(top_model)
    predictions = Dense(num_classes, activation='softmax', name='fc10')(top_model)
    resnet50_model_pluton = Model(inputs=X_input, outputs=predictions)

    return resnet50_model_pluton


def pre_trained_vgg16(input_shape, num_classes):
    model_vgg = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=input_shape
                       )
    for layer in model_vgg.layers:
        layer.trainable = False

    model = Flatten(name='flatten')(model_vgg.output)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(0.5)(model)
    model = Dense(num_classes, activation='softmax')(model)
    model_vgg_pluton = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pluton')

    return model_vgg_pluton

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

    # Create model
    Alex_model = Model(inputs=X_input, outputs=X, name='AlexNetNoTop')
    Alex_model.load_weights('E:\\datasets\\model\\alexnet_weights_pytorch_notop.h5')

    for layer in Alex_model.layers:
        layer.trainable = False

    model = Flatten(name='flatten')(Alex_model.output)

    model = Dense(1024, activation='relu', name='fc1024')(model)
    model = Dropout(0.5)(model)

    model = Dense(num_classes, activation='softmax')(model)
    model_Alex_pluton = Model(inputs=Alex_model.input, outputs=model, name='alexnet_pluton')

    return model_Alex_pluton


def lr_schedule(epoch):
    drop = 0.1
    epochs_drop = 10
    lr = lr_ini * pow(drop, floor((1 + epoch) / epochs_drop))
    print('Learning rate: ', lr)
    return lr


# Prepare callbacks for model saving and for learning rate adjustment.
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20, verbose=0, mode='max', baseline=None, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max',
                             period=1)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-9)

log_dir = './%slogs' % model_type
tbCallBack = TensorBoard(log_dir=log_dir,
                         histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         write_images=True,
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping, tbCallBack]

if model_type == 'pre_trained_VGG16':
    model = pre_trained_vgg16(input_shape, num_classes)
elif model_type == 'pre_trained_ResNet50':
    model = pre_trained_resnet50(input_shape, num_classes)
elif model_type == 'pre_trained_Xception':
    model = pre_trained_xception(input_shape, num_classes)
elif model_type == 'pre_trained_AlexNet':
    model = pre_trained_AlexNet(input_shape, num_classes)

# compile the model
optimizer = Adam(lr=lr_ini, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', acc_top2, acc_top3])
print("Model Summary of ", model_type)
print(model.summary())


if flow_from_dir:
    # generator for reading train data from folder
    print('Model fit using flow from directory')
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    # generator for reading validation data from folder
    validation_generator = datagen.flow_from_directory(
        validate_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=validation_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=epochs,
                                  callbacks=callbacks
                                  )

    scores = model.evaluate_generator(generator=validation_generator,
                                      steps=STEP_SIZE_VALID, verbose=1)