import argparse
import numpy as np
import pandas as pd

import bilinear
from keras import models, layers, utils, optimizers
from keras.applications import vgg16, vgg19
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from skimage import io, transform
from skimage.color import gray2rgb

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
parser.add_argument('--datapath', dest='datapath', default=None,
                    help='Path to root folder of dataset')
parser.add_argument('--fAfB', dest='modelstring', required=True,
                    help='Model string. One of '+l2s(valid_models))
parser.add_argument('--n_filters', dest='n_filters', default=None,
                    help='Reduce (or expand) output filters in fA/fB using 1x1 conv layer. Default=None')
parser.add_argument('--savepath', dest='savepath', default=None,
                    help='Path to save model to. Default=None')
args = parser.parse_args()

assert args.dataset in valid_datasets, 'dataset must be valid'
assert args.modelstring in valid_models, 'model (fAfB) must be valid'

###########################
#######   Dataset   #######
###########################
if args.dataset == 'birds':
    n_classes = 200
    input_shape = (448, 448, 3)

    dbox = '/home/administrator/Dropbox/benchmark/CUB_200_2011/npy/'
    X_train = np.load(dbox+'birds_X_train.npy')
    y_train = np.load(dbox+'birds_y_train.npy')
    X_test = np.load(dbox+'birds_X_test.npy')
    y_test = np.load(dbox+'birds_y_test.npy') 


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
def make_vgg(s, n_filters=None):
    if s == 'M':
        base = vgg16.VGG16(include_top=False)
        x = base.get_layer('block5_conv3').output
    elif s == 'D':
        base = vgg19.VGG19(include_top=False)
        x = base.get_layer('block5_conv4').output
        
    if n_filters is not None:
        x = layers.Conv2D(n_filters, 1, kernel_initializer='orthogonal', name='conv1x1')(x)

    return models.Model(inputs=base.input, outputs=x)

def make_vggBCNN(modelstring, input_shape, n_classes):
    n_filters = int(args.n_filters)
    if modelstring[0] == modelstring[1]:
        fA = make_vgg(modelstring[0], n_filters)
        fB = None
    else:
        fA = make_vgg(modelstring[0], n_filters)
        fB = make_vgg(modelstring[1], n_filters)

    return bilinear.combine(fA, fB, input_shape, n_classes)


model = make_vggBCNN(args.modelstring, input_shape, n_classes)
print("Created Model:")
print(model.summary())

# only train FC layer first
for layer in model.layers:
    not_pretrained =  'dense' in layer.name or 'conv1x1' in layer.name
    if not_pretrained:
        layer.trainable = True
    else:
        layer.trainable = False

nadam = optimizers.Nadam()
model.compile(optimizer=nadam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
print("Model Compiled")

#########################
#######   Train   #######
#########################
'''two step train: fine tune FC, then backprop w/ n~0.001, epochs=45-100'''

epochs = 100
batch_size = 16

train_flow = train_gen.flow(X_train, y_train, batch_size=batch_size)
test_flow = test_gen.flow(X_test, y_test, batch_size=batch_size)

early_stop = EarlyStopping(monitor='val_acc', patience=25)
callbacks  = [early_stop]

model.fit_generator(train_flow,
                    steps_per_epoch=len(X_train)/batch_size,
                    validation_data=test_flow,
                    validation_steps=len(X_test)/(batch_size*10),
                    callbacks=callbacks, verbose=1, epochs=epochs)

print('Finished training Dense layer...')
CHKPT_PATH = args.savepath + args.dataset + args.modelstring + '.hdf5'
model.save(CHKPT_PATH)

# now all the layers
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=nadam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

if args.savepath is not None:
    checkpoint = ModelCheckpoint(CHKPT_PATH, monitor='val_acc', save_best_only=True)
    callbacks.append(checkpoint)

history = model.fit_generator(train_flow,
                    steps_per_epoch=len(X_train)/batch_size,
                    validation_data=test_flow,
                    validation_steps=len(X_test)/(batch_size*10),
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

