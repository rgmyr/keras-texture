"""
My own experimental dilation-block networks for texture problems.
"""
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, MaxPool2D, Input, Lambda, Dense

#from texture.initializers import ConvolutionAware
from texture.networks.util import make_dense_layers
from texture.ops import bilinear_pooling

def dilation_branch(x, f_in, f_out, r, name='unnammed_conv'):
    '''
    Combination of 1x1 conv compression and 3x3 dilated conv expansion.
    '''
    x = Conv2D(f_in, (1,1), activation='relu', name=name+'_1x1_'+str(r))(x)

    #conv_init = ConvolutionAware()
    x = Conv2D(f_out, (3,3), padding='same', dilation_rate=(r,r), activation='relu', #kernel_initializer=conv_init,
               name=name+'_dilate_'+str(r))(x)

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
    block_output = Concatenate()(branch_outputs)

    if pool_size is not None:
        block_output = MaxPool2D(pool_size=pool_size)(block_output)

    block_output = BatchNormalization()(block_output)

    return block_output

'''
pooling_options = {
    'bilinear': bilinear_pooling
}


def auxiliary_pooling(x,
                      conv1x1=None,
                      pooling_type='bilinear',
                      output_size=256,
                      dropout_rate=None):
    Aux. pooling branch with `output_size` dense features.

    if conv1x1 is not None:
        x = Conv2D(conv1x1, (1,1), activation='relu')(x)

    x = pooling_options[pooling_type](x)

    x = make_dense_layers([output_size], dropout=dropout_rate)(x)

    return x
'''

def dilation_net(num_classes,
                 input_shape,
                 entry_conv=None,
                 dilation_rates=[1,2,3,4],
                 blocks=[32,48,64],
                 max_poolings=2,
                 conv1x1=None,
                 pooling_type='bilinear',
                 dense_layers=[],
                 dropout_rate=None):
    #CNN built on dilation blocks and auxillary poolings.
    if input_shape is None:
        input_shape = (None,None,3)

    input_image = Input(shape=input_shape)

    # Two 3x3 entry convs
    if entry_conv is not None:
        #conv_init = ConvolutionAware()
        x = Conv2D(int(entry_conv/2), (3,3), padding='same', activation='relu', # kernel_initializer=conv_init,
                   name='entry_3x3_1')(input_image)
        x = Conv2D(entry_conv, (3,3), padding='same', activation='relu', # kernel_initializer=conv_init,
                   name='entry_3x3_2')(x)
    else:
        x = input_image

    if not hasattr(dilation_rates[0], '__iter__'):
        dilation_rates = [dilation_rates]*len(blocks)
        print(dilation_rates)

    block_poolings = []
    for i, (f, branches) in enumerate(zip(blocks, dilation_rates)):
        max_pool = (2,2) if i < max_poolings else None
        x = dilation_block(x, branches, f, f, pool_size=max_pool, name='block_'+str(f))
        reduce_size = conv1x1 if conv1x1 is not None else f
        reduce = Conv2D(reduce_size, (1,1), activation='relu')(x)
        pooling = Lambda(bilinear_pooling, name='bilinear_pooling_'+str(f))([reduce, reduce])
        block_poolings.append(pooling)

    x = Concatenate()(block_poolings)  # normalize here?

    x = make_dense_layers(dense_layers, dropout=dropout_rate)(x)

    pred = Dense(num_classes, activation='softmax')(x)

    model = KerasModel(inputs=input_image, outputs=pred)

    return model
