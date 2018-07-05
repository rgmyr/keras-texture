import argparse
import numpy as np
import pandas as pd

import bilinear
from keras import models
from keras import utils
from keras import optimizers
from keras.applications import vgg16, vgg19
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from skimage import io, transform

#########################
#######    Args   #######
#########################
valid_models   = ['MM','MD','DD']
valid_datasets = ['birds'] #,'cars','airplanes']

def l2s(slist):
    return '{'+','.join(slist)+'}'

parser = argparse.ArgumentParser(description='Train B-CNN built on VGG-M/D')
parser.add_argument('--dataset', dest='dataset', required=True,
                    help='Name of benchmark dataset. One of '+l2s(valid_datasets))
parser.add_argument('--datapath', dest='datapath', required=True,
                    help='Path to root folder of dataset')
parser.add_argument('--fAfB', dest='modelstring', required=True,
                    help='Model string. One of '+l2s(valid_models))
parser.add_argument('--savepath', dest='savepath', default=None,
                    help='Path to save model to. Default=None')
args = parser.parse_args()

assert args.dataset in valid_datasets, 'dataset must be valid'
assert args.modelstring in valid_models, 'model (fAfB) must be valid'

###########################
#######   Dataset   #######
###########################
def crop_and_resize(img, resize_shape):
    h, w, _ = img.shape
    diff = np.abs(h-w)
    start = diff // 2
    if h > w:
        img = img[start:-start,:,:]
    elif w > h:
        img = img[:,start:-start,:]
    return transform.resize(img, resize_shape)


if args.dataset == 'birds':
    n_classes = 200
    input_shape = (448, 448, 3)

    img_dir = args.datapath + 'images/'
    img_paths = img_dir + pd.read_csv(args.datapath+'images.txt', index_col=0, header=None, sep=' ')[1]
    img_paths = img_paths.values
    print(img_paths)

    img_labels = np.loadtxt(args.datapath+'image_class_labels.txt',dtype=np.uint8)[:,1]

    idxs = np.loadtxt(args.datapath+'train_test_split.txt',dtype=np.uint8)[:,1]
    train_ix = np.where(idxs == 1)
    test_ix  = np.where(idxs == 0)

    X_train, y_train = np.empty((len(train_ix), 448, 448, 3)), np.zeros(len(train_ix))
    X_test , y_test  = np.empty((len(test_ix) , 448, 448, 3)), np.zeros(len(test_ix))

    for i, idx in enumerate(list(train_ix)):
        img = io.imread(img_paths[idx])
        X_train[i,...] = crop_and_resize(img, input_shape)
        y_train[i] = img_labels[idx]
    for i, idx in enumerate(list(test_idx)):
        img = io.imread(img_paths.iloc[idx])
        X_test[i,...] = crop_and_resize(img, input_shape)
        y_test[i] = img_labels[idx]

    y_train = utils.to_categorical(y_train, n_classes)
    y_test = utils.to_categorical(y_test, n_classes)


'''Preprocess: crop central square and resize to 448x448.'''
train_gen = ImageDataGenerator(width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.05,
                               horizontal_flip=True)
test_gen = ImageDataGenerator()

train_gen.fit(X_train)
test_gen.fit(X_test)

#########################
#######   Model   #######
#########################
def make_vgg(s):
    if s == 'M':
        base = vgg16.VGG16(include_top=False)
        output = base.get_layer('block5_conv3').output
    elif s == 'D':
        base = vgg19.VGG19(include_top=False)
        output = base.get_layer('block5_conv4').output

    return models.Model(inputs=base.input, outputs=output)

def make_vggBCNN(modelstring, input_shape, n_classes):
    if modelstring[0] == modelstring[1]:
        fA = make_vgg(modelstring[0])
        fB = None
    else:
        fA = make_vgg(modelstring[0])
        fB = make_vgg(modelstring[1])

    return bilinear.combine(fA, fB, input_shape, n_classes)


model = make_vggBCNN(args.modelstring,)
print("Created Model:")
print(model.summary())

# only train FC layer first
for layer in model.layers:
    if 'dense' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

rms = optimizers.RMSprop()
model.compile(optimizer=rms,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
print("Model Compiled")

#########################
#######   Train   #######
#########################
'''two step train: fine tune FC, then backprop w/ n~0.001, epochs=45-100'''

epochs = 100
batch_size = 32

train_flow = train_gen.flow(X_train, y_train, batch_size=batch_size)
test_flow = test_gen.flow(X_test, y_test, batch_size=batch_size)

early_stop = EarlyStopping(monitor='val_acc', patience=25)
callbacks  = [early_stop]

model.fit_generator(train_flow,
                    steps_per_epoch=len(X_train)/batch_size,
                    validation_data=test_flow,
                    validation_steps=len(X_test)/batch_size,
                    callbacks=callbacks, verbose=1, epochs=epochs)

print('Finished training Dense layer...')

# now all the layers
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=rms,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

if args.savepath is not None:
    CHKPT_PATH = args.savepath + args.dataset + args.modelstring + '.hdf5'
    checkpoint = ModelCheckpoint(CHKPT_PATH, monitor='val_acc', save_best_only=True)
    callbacks.append(checkpoint)

history = model.fit_generator(train_flow,
                    steps_per_epoch=len(X_train)/batch_size,
                    validation_data=test_flow,
                    validation_steps=len(X_test)/batch_size,
                    callbacks=callbacks, verbose=1, epochs=epochs)

if args.savepath is not None: 
    ppath = args.savepath + args.dataset + args.modelstring + '_HISTORY_FULL'
    with open(ppath, 'wb') as hist_pickle:
        pickle.dump(history.history, hist_pickle)

print("Finished training model...")

preds = model.predict_generator(test_gen.flow(X_test))
preds = np.argmax(preds, axis=1)
ytrue = np.argmax(ytrue, axis=1)

print("Test accuracy: ", sum(np.equal(ytrue,preds))/ytrue.size)

