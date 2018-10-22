"""
My own experimental dilation-block networks for texture problems.
"""
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, MaxPool2D, Input, Lambda, Dense

from texture.networks.util import make_dense_layers, make_pooling_layer, auxiliary_pooling
from texture.layers import DARTSEdge


def dilation_branch(x, f_in, f_out, r, strides=1, name='unnammed_conv'):
    """
    Combination of 1x1 conv compression and 3x3 dilated conv expansion.
    """
    x = Conv2D(f_in, (1,1), strides=strides, activation='relu', name=name+'_1x1_'+str(r))(x)

    x = Conv2D(f_out, (3,3), padding='same', dilation_rate=(r,r), activation='relu', name=name+'_dilate_'+str(r))(x)

    return x


def dilation_block(x,
                   branches,
                   in_filters,
                   out_filters,
                   strides=1,
                   combine_mode='concat',
                   batchnorm=True,
                   name='block'):
    """Block with branches defined by dilation rates of `branches`.

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
    strides : int, optional
        Strides (in x and y) for 1x1 reduction conv, default=1.
    combine_mode : str, optional
        If `DARTS` found in combine_mode.upper(), use DARTSEdge. Otherwise, default=Concatenate.
    name : str, optional
        Base name for block/branch layer names

    Returns
    -------
    Output of passing `x` through the constructed block.
    """

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

    # Branch definition
    branch_outputs = []
    for rate, f_in, f_out in zip(branches, in_filters, out_filters):
        branch = dilation_branch(x, f_in, f_out, rate, strides=strides, name=name)
        branch_outputs.append(branch)

    # Branch combination
    if 'DARTS' in combine_mode.upper():
        block_output = Lambda(lambda x: tf.stack(x, axis=1))(branch_outputs)
        block_output = DARTSEdge(name=name+'_DARTS')(block_output)
    else:
        block_output = Concatenate()(branch_outputs)

    if batchnorm:
        block_output = BatchNormalization()(block_output)

    return block_output


def diception_net(num_classes,
                 input_shape,
                 entry_conv=None,
                 dilation_rates=[1,2,3,4],
                 blocks=[32,48,64],
                 expansion_factor=1.0,
                 combine_mode='concat',
                 strides=2,
                 pooling_args={'bilinear': {'conv1x1': 32}},
                 pooling_features=256,
                 dense_layers=[],
                 dropout_rate=None):
    """Dilated inception-like network for texture recognition, optionally
    with DARTS for differentiable dilation rate & pooling method search.

    Parameters
    ----------
    num_classes : int
    input_shape : tuple(int)
    entry_conv : int, optional
        If given, network starts with two 3x3 convs with size: (3->entry_conv//2) + (entry_conv//2->entry_conv).
    dilation_rates : list(int) or list(list(int)), optional
        Dilation rates of branches in each block. If list with depth of one, same set of rates used for each block.
    blocks : list(int), optional
        Output size of all branches in a given block. Note: branches are concatenated unless `combine_mode=darts`.
    reduction_factor : numeric, optional
        Defines branch input reduction as block_size // reduction_factor, default=1. Default makes sense for non-DARTS,
        need to think about what to do for DARTS combine.
    strides : int, optional
        Downsampling occurs in 1x1 branch entry convs via striding, default=2 (same for all blocks).
    pooling_args : dict, optional
        Dict specify auxiliary_pooling method(s) like {'method_name': {<dict of kwargs>}}. If more than one method is given,
        results are combined using DARTSEdge and named appropriately in edge.op_names.
    pooling_features : int, optional
        Pooling output(s) are reduced by a dense layer with `pooling_features` output nodes, default=256.
    ...
    """
    #CNN built on dilation blocks and auxillary poolings.
    if input_shape is None:
        input_shape = (None, None, 3)

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
        f_in = int(f // expansion_factor)
        x = dilation_block(x, branches, f_in, f, strides=strides, combine_mode=combine_mode, name='block_'+str(f))

        pooling_ops = []
        for pname, pargs in pooling_args.items():
            pargs['name'] = 'block_'+str(f)
            p = auxiliary_pooling(x, pname, output_size=pooling_features, dropout=dropout_rate, **pargs)
            pooling_ops.append(p)

        if len(pooling_ops) > 1:
            pooling = Lambda(lambda x: tf.stack(x, axis=1))(pooling_ops)
            pooling = DARTSEdge(op_names=list(pooling_args.keys()), name='block_'+str(f)+'_poolingDARTS')(pooling)
        else:
            pooling = pooling_ops[0]

        block_poolings.append(pooling)

    x = Concatenate()(block_poolings)  # normalize here?

    x = make_dense_layers(dense_layers, dropout=dropout_rate)(x)

    pred = Dense(num_classes, activation='softmax')(x)

    model = KerasModel(inputs=input_image, outputs=pred)

    return model
