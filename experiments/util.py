from time import time

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


def train_model(model, dataset, epochs, batch_size, checkpoint=False, gpu_ind=None):
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)
        callbacks.append(early_stopping)
    # GPU_UTIL_SAMPLER

    if checkpoint:
        model_checkpoint = ModelCheckpoint(model.weights_filename, monitor='val_loss', verbose=1, 
                                          save_best_only=True, save_weights_only=True)
        callbacks.append(model_checkpoint)

    model.network.summary()

    t = time()
    history = model.fit(dataset, batch_size, epochs, callbacks)
    print('Training took {:2f} s'.format(time() - t))

    return model
