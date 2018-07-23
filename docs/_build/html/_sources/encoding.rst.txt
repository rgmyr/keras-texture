Encoding Models
===============

Encoding-based models are those that use a codebook of codeword vectors to compress the output volume of a CNN.

Some approaches learn these codewords in an end-to-end trainable layer (or set of layers). These include ``DeepTEN``, and the ``DEP`` network, which use the ``Encoding`` and ``BilinearModel`` layers.

In other (typically earlier) approaches, the codewords are generated from the distribution of pretrained CNN feature vectors on a the dataset of interest. This package includes the Fisher vector variant of this approach (``FV-CNN``), since it is relatively simple and usually outperforms similar encoding schemes (*e.g.*, ``VLAD``, ``bag-of-words``). 


``Encoding`` Layer
------------------

The residual encoding layer proposed in `Deep TEN: Texture Encoding Network <https://arxiv.org/pdf/1612.02844.pdf>`__ [*CVPR*, 2017]:

.. figure:: ./images/Encoding-Layer_diagram.png
   :alt: Encoding-Layer

The layer learns a ``KxD`` dictionary of codewords (a "codebook"), and codeword assignment ``scale`` weights. These are used to encode the residuals of an input of shape ``NxD`` or ``HxWxD`` with respect to the codewords. 

Let :math:`X = \{x_1,...,x_N\}` be an set of ``N`` input feature vectors of length ``D``, and :math:`C = \{c_1,...,c_K\}` be the set of ``K`` codewords of length ``D``. The elements of the aggregate residual encoding :math:`E = \{e_1,...,e_K\}` are given by:
    
.. math::
    e_j = \sum_{i=1}^{N} w_{ij}r_{ij}

Where residual vectors :math:`r_ij = x_i - c_j`. The set of weights for a given input :math:`x_i` is the codeword-wise softmax of residual vector norms scaled by the learnable smoothing factors :math:`\{s_1,...,s_K\}`:

.. math::
    w_{ij} = \frac{\exp(-s_j||r_{ij}||^2)}{\sum_{k=1}^{K}\exp(-s_k||r_{ik}||^2)}

This ``keras`` implementation is largely based on the `PyTorch-Encoding <https://github.com/zhanghang1989/PyTorch-Encoding>`__ release by the paper authors. It includes optional L2 normalization of output vectors (``True`` by default) and dropout (``None`` by default). Unlike the ``PyTorch-Encoding`` version, only the number of codewords ``K`` needs to be specified at construction time -- the feature size ``D`` is inferred from the ``input_shape``.


``BilinearModel`` Layer
-----------------------

``texture.layers.BilinearModel`` is a trainable ``keras`` layer implementing the weighted outer product of inputs with shape ``[(batches,N),(batches,M)]``. The original idea of bilinear modeling for computer vision problems dates back to `Learning Bilinear Models for Two-Factor Problems in Vision <http://www.merl.com/publications/docs/TR96-37.pdf>`__ [*CVPR*,1997].

It is used in the ``Deep Encoding Pooling Network (DEP)`` proposed in `Deep Texture Manifold for Ground Terrain Recognition <https://arxiv.org/abs/1803.10896>`__ [*CVPR*, 2018] to merge the output of an ``Encoding`` layer with the output of a standard global average pooling, where both features are extracted from the convolutional output of the same ``ResNet`` base. 

.. figure:: ./images/DEP_diagram.png
   :alt: DEP-Architecture

The intuition is that the former represents textures (via orderless encoding) and the latter represents spatially structured observations, so that "*[the] outer product representation captures a pairwise correlation between the material texture encodings and spatial observation structures.*"


``FVCNN`` Model
---------------

The Fisher vector encoding of CNN features was proposed in `Deep filter banks for texture recognition and segmentation <https://www.robots.ox.ac.uk/~vgg/publications/2015/Cimpoi15/cimpoi15.pdf>`__ [2015]. A Fisher vector encoding is parametrized by a Gaussian Mixture Model of the feature vector distribution comprised of ``K`` Gaussians. The Fisher vector representation of a ``NxD`` set of local feature vectors is the concatenation of the channel-wise deviances in mean and variance between the input vectors and the ``K`` Gaussians. 

The ``texture.fisher.FVCNN`` class provides a wrapper for fitting a GMM to a training set given a CNN (can be from ``keras.applications``, or an arbitrary model), generating Fisher vector encodings, and training a SVM classifier.

