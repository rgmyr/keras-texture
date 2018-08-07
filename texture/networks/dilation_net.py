"""My own experimental dilation-block models for texture.

"""
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, MaxPool2D

from texture.initializers import ConvolutionAware


def dilation_branch(x, f_in, f_out, r, pool=False, name='unnammed_conv'):
    '''Combination of 1x1 conv compression and 3x3 dilated conv expansion.
    '''
    x = Conv2D(f_in, activation='relu', name=name+'_1x1_'+str(r))(x)
    x = BatchNormalization()(x)

    conv_init = ConvolutionAware()
    x = Conv2D(f_out, (3,3), padding='same', dilation_rate=(r,r), activation='relu',
              initializer=conv_init, name=name+'_dilate_'+str(r))(x)
    x = BatchNormalization()(x)

    return x


def dilation_block(x,
                   branches,
                   in_filters,
                   out_filters,
                   pool_size=None,
                   name='block'):
    '''Block with branches defined by dilation rates of `branches`.

    Parameters
    ----------
    x : Tensor, BxHxWxC
        Input Tensor
    branches : iterable(int)
        Each item defines a branch with dilation_rate=item
    in_filters : iterable or int
        Output filters for 1x1 reduction, per branch.
        If iterable, must have len == len(branches). If int, assumed all branches the same.
    out_filters : iter
        Output filters for 3x3 dilated conv, per branch.
        If iterable, must have len == len(branches). If int, assumed all branches the same.
    pool_size : tuple(int), or `None`
        If not `None`, passed to MaxPool2D on block output.
    name : str, optional
        Base name for block/branch layer names

    Returns
    -------
    Output of passing `x` through the constructed block.
    '''

    args = [branches, in_filters, out_filters]
    if all([hasattr(f, '__iter__') for f in args]):
        print([lambda f: hasattr(f, '__iter__') for f in args])
        assert len(set([len(x) for x in args])) == 1, \
            'if iterables, in_filters & out_filters must have same len as branches'
    elif isinstance(in_filters, int) and isinstance(out_filters, int):
        # if both ints
        in_filters = [in_filters] * len(branches)
        out_filters = [out_filters] * len(branches)
    else:
        raise ValueError('in_filters & out_filters must be both `iterable`, or both `int`')

    branch_outputs = []
    for rate, f_in, f_out in zip(branches, in_filters, out_filters):
        branch = dilation_branch(x, f_in, f_out, rate, name=name)
        branch_outputs.append(branch)
    block_output = Concatenate()(branches_out)

    if pool_size is not None:
        block_output = MaxPool2D(pool_size=pool_size)(block_output)

    return block_output


def dilation_net(input_shape=None):
    '''CNN built on dilation blocks and auxillary poolings.

    '''
    input_image = Input(shape=input_shape)

    for


    model = KerasModel(inputs=input_image, outputs=pred)

    return model
