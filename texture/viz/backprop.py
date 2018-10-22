"""
https://distill.pub/2017/feature-visualization/
"""



def maximize_class_idx(model, idx, step=1., threshold=0.999, max_iters=100):
    """Find an input that maximizes a given class probability with confidence > threshold."""
    loss = K.mean(model.output[:,idx])

    grads = K.gradients(loss, model.input)[0]

    # normalize the grads
    grads /= K.sqrt(K.mean(K.square(grads)) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    input_data = np.random.random((1)+model.input_shape[1:])

    for i in range(max_iters):
        loss_val, grads_val = iterate([input_data])
        input_data += grads_val * step

        print('Current confidence: {}'.format(loss_val))
        if loss_val > threshold:
            break

    return input_data
