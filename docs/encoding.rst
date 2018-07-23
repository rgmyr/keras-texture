``Encoding`` Layer
==================

The residual encoding layer proposed in `Deep TEN: Texture Encoding Network <https://arxiv.org/pdf/1612.02844.pdf>`__ [*CVPR*, 2017]:

.. figure:: ./images/Encoding-Layer_diagram.png
   :alt: Encoding-Layer

The layer learns a ``KxD`` dictionary of codewords (a "codebook"), and codeword assignment ``scale`` weights. These are used to encode the residuals of an input of shape ``NxD`` or ``HxWxD`` with respect to the codewords. 

Let :math:`X = \{x_1,...,x_n\}` be an set of ``N`` input feature vectors of length ``D``, and :math:`C = \{c_1,...,c_k\}` be the set of ``K`` codewords of length ``D``. The residual encodings :math:`E = \{e_1,...,e_k\}` are computed as:
    
.. math::
    e_j = \sum_{i=1}^{N} w_{ij}r_{ij}

Where residual vectors :math:`r_ij = x_i - c_j`, and the weights depend on smoothing/scale factors :math:`\{s_i,...,s_k\}`, and are given by:

.. math::
    w_ij = \frac{\exp(-s_j||r_{ij}||^2)}{\sum_{k=1}^{K}\exp(-s_k||r_ik||^2)}


Implementation Notes
--------------------

This ``keras`` implementation is largely based on the `PyTorch-Encoding <https://github.com/zhanghang1989/PyTorch-Encoding>`__ release by the paper authors. It includes optional L2 normalization of output vectors (``True`` by default) and dropout (``None`` by default). Unlike the ``PyTorch-Encoding`` version, only the number of codewords ``K`` needs to be specified at construction time -- the feature size ``D`` is inferred from the ``input_shape``.




