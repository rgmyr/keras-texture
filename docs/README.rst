**TODO**:

- Weight initializers

   - Logistic Regression init. for ``softmax`` layers (esp. for large layers -- e.g., bilinear pooling output)
   - Try `ConvolutionAware <https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/initializers/convaware.py>`__ for CNNs?

- Expand base CNN builders

   - Convert weights from ResNet18/34 pre-trained ``Caffe`` models
   - Check out wide/dilated ResNet blocks from `keras-contrib/applications <https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications>`__

- Implement *compact* bilinear pooling

- Figure out Buffer bugs when passing ``covariance_bound`` to ``cyvlfeat.gmm.gmm``
- Tests and benchmarks

keras-texture
=============

Implementations of several ``keras`` layers, model classes, and other
utilities that are useful in constructing models for texture recognition
and fine-grained classification problems. It is a **work in progress**,
and the ``tensorflow`` backend is required for most functionality.

Develop-mode installable with ``pip install -e .`` Root module of package is ``texture``.


Benchmarks
----------

Working on benchmarking models constructed with various texture
recognition datasets:

Some fine-grained classification datasets are also of interest, but
benchmarking those has a lower priority for me at the moment:

-  `Birds-200 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`__
   (2011 version)
-  `FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`__
-  `Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`__


Further Improvements
--------------------

Encoding
^^^^^^^^

-  Smaller ``ResNet``-based constructors for feature networks

Bilinear
^^^^^^^^

-  Add support for ``fA`` and ``fB`` to have different input shapes
   (technically only output shapes need to correspond).
-  Add support for ``fA`` and ``fB`` to have different output shapes
   (crop/interpolate/pool to match them)

Would also like to add the matrix square root normalization layer as
described in:

::

    @inproceedings{lin2017impbcnn,
        Author = {Tsung-Yu Lin, and Subhransu Maji},
        Booktitle = {British Machine Vision Conference (BMVC)},
        Title = {Improved Bilinear Pooling with CNNs},
        Year = {2017}}

Authors claim this improves accuracy by several % on fine-grained
recognition benchmarks.

DEP
^^^

-  Utilities for combining a base CNN with ``Encoding`` & ``BilinearModel`` to create a ``Deep Encoding Pooling Network``.
