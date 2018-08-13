from time import time
import numpy as np
import os

#from clr_callback import CyclicLR
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler

# for LR_RAMP experiments
lr_low = 10e-7
lr_high = 0.05


def simple_cyclic_lr(low, high, step_size):
    pass

def train_model(model,
                dataset,
                epochs,
                batch_size,
                flags,
                gpu_ind=None):
    callbacks = []
    save_dir = os.path.join('/home/'+os.environ['USER']+'/Dropbox/benchmark', model.name)

    if 'EARLY_STOPPING' in flags:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1)
        callbacks.append(early_stopping)
    # GPU_UTIL_SAMPLER

    if 'LR_RAMP' in flags:    # run an increasing LR experiment
         lr = np.linspace(lr_low, lr_high, num=epochs)
         lr_schedule = LearningRateScheduler(lambda epoch: lr[epoch], verbose=1)
         print('Running LR_RAMP experiment with range ', (lr_low, lr_high))
         save_dir = os.path.join(save_dir, 'lr_ramp')
         callbacks.append(lr_schedule)

    if 'TENSORBOARD' in flags:
        tensorboard = TensorBoard(log_dir=save_dir, batch_size=batch_size)
        print('Saving TENSORBOARD events to ', save_dir)
        callbacks.append(tensorboard)

    if 'CHECKPOINT' in flags:
        model_checkpoint = ModelCheckpoint(model.weights_filename(save_dir), monitor='val_loss', verbose=1,
                                           save_best_only=True, save_weights_only=True)
        callbacks.append(model_checkpoint)

    #model.network.summary()

    t = time()
    history = model.fit(dataset, batch_size, epochs, callbacks)
    print('Training took {:2f} s'.format(time() - t))

    return model
