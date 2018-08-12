from time import time
import numpy as np
import os

#from clr_callback import CyclicLR
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler

root_dir = 'home/'+os.environ['USER']'/Dropbox/benchmark/'

# for ramp experiments
lr_low = 5e-4
lr_high = 5e-1


def simple_cyclic_lr(low, high, step_size):
    pass

def train_model(model,
                dataset,
                epochs,
                batch_size,
                flags,
                gpu_ind=None):
    callbacks = []

    if 'EARLY_STOPPING' in flags:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)
        callbacks.append(early_stopping)
    # GPU_UTIL_SAMPLER

    if 'CHECKPOINT' in flags:
        model_checkpoint = ModelCheckpoint(model.weights_filename, monitor='val_loss', verbose=1,
                                          save_best_only=True, save_weights_only=True)
        callbacks.append(model_checkpoint)

    if 'LR_RAMP' in flags:    # run a long increase LR experiment
         lr = np.linspace(lr_low, lr_high, num=epochs)
         lr_schedule = LearningRateScheduler(lambda epoch: lr[epoch])
         callbacks.append(lr_schedule)

    if 'TENSORBOARD' in flags:
        tensorboard = TensorBoard(log_dir=log_dir+model.name, batch_size=batch_size)
        callbacks.append(tensorboard)

    if 'CHECKPOINT' in flags:
        weights_dir = model.weights_filename(log_dir+model.name)
        model_checkpoint = ModelCheckpoint(model.weights_filename, monitor='val_loss', verbose=1,
                                          save_best_only=True, save_weights_only=True)
        callbacks.append(model_checkpoint)

    #model.network.summary()

    t = time()
    history = model.fit(dataset, batch_size, epochs, callbacks)
    print('Training took {:2f} s'.format(time() - t))

    return model
