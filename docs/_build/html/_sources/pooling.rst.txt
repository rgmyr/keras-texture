Pooling Models
==============

TODO...

Bilinear ``pooling``
--------------------

``bilinearpooling.py`` provides a few convenience functions for creating
symmetric or asymmetric B-CNN models in Keras with bilinear pooling, as
proposed in `Bilinear CNNs for Fine-grained Visual
Recognition <http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf>`__
(*ICCV*, 2015).

``bilinearpooling.pooling``:

-  Average pooling of local feature vector outer products in
   ``tensorflow``
-  Includes element-wise signed square root and L2 normalization
-  If using ``combine``, you won't need to reference this explicitly

``bilinearpooling.combine``:

-  Takes two ``keras`` models ``fA`` and ``fB`` with output shapes
   ``(N, H, W, cA)``, ``(N, H, W, cB)``
-  Maps ``[fA.output, fB.output]`` to shape ``(N, cA, cB)`` with
   ``bilinear.pooling``
-  Flattens, connects to ``softmax`` output using a specifiable number
   of ``Dense`` layers.
-  Returns the resulting ``keras.models.Model`` instance

Usage Notes
^^^^^^^^^^^

-  Be careful with reuse of single model for ``fA`` and ``fB`` (*e.g.*,
   asymmetry via different output layers). Weights will be shared if you
   use the same instantiation of the original model to generate both
   models.

If the dimensionality of local feature vectors is 512, and there are
``N`` classes, the size of a fully-connected classification layer will
be very large (``512*512*N=262,144*N``). With random weight
initialization, it seems pretty difficult to train a layer of this size
for moderate to large ``N``, so I'm looking at writing an initializer
that uses logistic regression, something which is *not* mentioned in the
paper, but which is present in the authors' matlab release.

``KernelPooling`` Layer
-----------------------

Implementation of `Kernel Pooling for Convolutional Neural
Networks <https://vision.cornell.edu/se3/wp-content/uploads/2017/04/cui2017cvpr.pdf>`__
[*CVPR*, 2017]. The layer uses the Count Sketch projection to compute a
*p*-order Taylor series kernel with learnable composition. The
composition weights *alpha* are initialized to approximate a Gaussian
RBF kernel. The kernel is computed over all local feature vectors
``(h_i, w_j)`` in the input volume and then average pooled.

.. figure:: ./images/kernel_pooling_diagram.png
   :alt: Kernel-Pooling

Construction paramters include ``p`` (order of the kernel
approximation), ``d_i`` (dimensionality for each order ``i>=2``). Output
has shape ``(batches, 1+C+(p-1)*d_i)``, where ``C`` is the number of
input channels.

The *gamma* parameter, which determines *alpha* values in the
approximation under the assumption of L2-normalized input vectors, can
optionally be estimated using a set of training feature vectors.



