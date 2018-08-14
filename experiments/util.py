from time import time
import numpy as np
import os

#from clr_callback import CyclicLR
from tensorflow.keras.backend import eval
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler

# for LR_RAMP experiments --> log(lr)
lr_low = -6
lr_high = 0


# CylicLR seems to not actually update LR, so simple triangular subsitute:
def simple_cyclic_lr(base_lr=0.0001, max_lr=0.1, step=5):
    '''Return func[epoch --> lr]. Step given in epochs.'''
    def lr_getter(epoch):
        cycle = np.floor(1+epoch/(2*step))
        x = np.abs(epoch/step - 2*cycle + 1)
        return base_lr + (max_lr-base_lr)*np.maximum(0,(1-x)) #*scale_fn(epoch)
    return lr_getter


class LRTensorBoard(TensorBoard):
    '''TensorBoard subclass to add LR scalar to events log.'''
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def train_model(model,
                dataset,
                epochs,
                batch_size,
                flags,
                flag_args,
                gpu_ind=None,
                save_ext=None):
    callbacks = []
    assert not ('LR_RAMP' in flags and 'CYCLIC_LR' in flags), 'Only one LR flag allowed per experiment'
    LR_IS_SCHEDULED = ('LR_RAMP' in flags or 'CYCLIC_LR' in flags)


    save_dir = os.path.join('/home/'+os.environ['USER']+'/Dropbox/benchmark', model.name)
    if save_ext is not None:
        save_dir = os.path.join(save_dir, save_ext)

    if 'EARLY_STOPPING' in flags:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=25, verbose=1)
        callbacks.append(early_stopping)
    # GPU_UTIL_SAMPLER

    if 'LR_RAMP' in flags:    # run an increasing LR experiment
         save_dir = os.path.join(save_dir, 'lr_ramp')
         lr_low = flag_args.pop('lr_low', -6)
         lr_high = flag_args.pop('lr_high', 0)
         lr = np.logspace(lr_low, lr_high, num=epochs)
         lr_ramp = LearningRateScheduler(lambda epoch: lr[epoch], verbose=1)
         print('Running LR_RAMP experiment with log range ', (lr_low, lr_high))
         callbacks.append(lr_ramp)

    if 'CYCLIC_LR' in flags:
        save_dir = os.path.join(save_dir, 'cyclic_lr')
        lr_getter = simple_cyclic_lr(**flag_args)
        lr_cyclic = LearningRateScheduler(lr_getter, verbose=1)
        print('Running CYCLIC_LR experiment with params: ', flag_args)
        callbacks.append(lr_cyclic)

    if 'TENSORBOARD' in flags:
         lr_tensorboard = LRTensorBoard(log_dir=save_dir)
         print('Logging TENSORBOARD at ', save_dir)
         callbacks.append(lr_tensorboard)

    if 'CHECKPOINT' in flags:
        model_checkpoint = ModelCheckpoint(model.weights_filename(save_dir), monitor='val_loss', verbose=1,
                                           save_best_only=True, save_weights_only=True)
        callbacks.append(model_checkpoint)

    #model.network.summary()

    t = time()
    history = model.fit(dataset, batch_size, epochs, callbacks)
    print('Training took {:2f} s'.format(time() - t))

    return model
