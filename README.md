# keras-texture

Implementations of several `tf.keras` layers, model classes, and other utilities that are useful in constructing  and training models for texture recognition and fine-grained classification problems. 

~~It is a **work in progress**.~~ Actually, that's generous. I stopped developing this project awhile ago. I do intend to eventually come back and clean things up, but it will probably be most useful to you if you're willing to fork it and do a bit of hacking as necessary. Still, I hope it's a useful starting point and I will try to help with any sticking points if asked.

Develop-mode installable with `pip install -e .` Root module of package is `texture`.

**TODO**

- Clean up notebooks + experiments
- Implement *compact* bilinear pooling
- More experiments + tweaking to get closer to claimed performance levels
- Figure out Buffer bugs when passing `covariance_bound` to `cyvlfeat.gmm.gmm` (created issue, no responses.)

## Requirements

- `numpy`
- `scikit-image`
- `scikit-learn`
- `tensorflow`

The TensorFlow requirement is not enforced in `setup.py` due to the ambiguity between `tensorflow` and `tensorflow-gpu`. This package allows CPU or GPU versions, since some functionality (*e.g.*, Fisher vector encoding with pretrained models) don't necessarily require a GPU.

#### Additional requirements: FV-CNN

Use of the Fisher vector CNN class (`texture.models.FVCNN`) requires the [cyvlfeat](https://github.com/menpo/cyvlfeat) wrappers for VLFeat, which should be installed using conda: `conda install -c menpo cyvlfeat`, if at all possible. It also requires `scikit-learn`, particularly the `svm.LinearSVC` class.

Neither of these packages are required in other `texture` modules, so they are not explicitly enforced in `setup.py`.

# Contents

## `Encoding` Layer

The residual encoding layer proposed in [Deep TEN: Texture Encoding Network](https://arxiv.org/pdf/1612.02844.pdf) [*CVPR*, 2017]. This `keras` implementation is largely based on the [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) release by the paper authors.

<p align="center">
  <img src="./docs/images/Encoding-Layer_diagram.png?raw=true" alt="Encoding-Layer diagram"/>
</p>

The layer learns a `KxD` dictionary of codewords (a "codebook"), and codeword assignment `scale` weights. These are used to encode the residuals of an input of shape `NxD` or `HxWxD` with respect to the codewords. Includes optional L2 normalization of output vectors (`True` by default) and dropout (`None` by default). Unlike the `PyTorch-Encoding` version, only the number of codewords `K` needs to be specified at construction time -- the feature size `D` is inferred from the `input_shape`.

## `BilinearModel` Layer

`BilinearModel` is a trainable `keras` layer implementing the weighted outer product of inputs with shape `[(batches,N),(batches,M)]`. The original idea of bilinear modeling for computer vision problems was proposed in [Learning Bilinear Models for Two-Factor Problems in Vision](http://www.merl.com/publications/docs/TR96-37.pdf) [*CVPR*, 1997].

It is used in the `Deep Encoding Pooling Network (DEP)` proposed in [Deep Texture Manifold for Ground Terrain Recognition](https://arxiv.org/abs/1803.10896) [*CVPR*, 2018] to merge the output of an `Encoding` layer with the output of a standard global average pooling, where both features are extracted from `conv` output of the same `ResNet` base. The intuition is that the former represents textures (orderless encoding) and the latter represents spatially structured observations, so that "[the] outer product representation captures a pairwise correlation between the material texture encodings and spatial observation structures."

![DEP-Architecture](./docs/images/DEP_diagram.png)

## `KernelPooling` Layer

Implementation of [Kernel Pooling for Convolutional Neural Networks](https://vision.cornell.edu/se3/wp-content/uploads/2017/04/cui2017cvpr.pdf) [*CVPR*, 2017]. The layer uses the Count Sketch projection to compute a *p*-order Taylor series kernel with learnable composition. The composition weights *alpha* are initialized to approximate a Gaussian RBF kernel. The kernel is computed over all local feature vectors `(h_i, w_j)` in the input volume and then average pooled.

<p align="center">
  <img src="./docs/images/kernel_pooling_diagram.png?raw=true" alt="Kernel Pooling"/>
</p>

Construction paramters include `p` (order of the kernel approximation), `d_i` (dimensionality for each order `i>=2`). Output has shape `(batches, 1+C+(p-1)*d_i)`, where `C` is the number of input channels.

The *gamma* parameter, which determines *alpha* values in the approximation under the assumption of L2-normalized input vectors, can optionally be estimated using a set of training feature vectors.

## Bilinear `pooling`

`bilinearpooling.py` provides a few convenience functions for creating symmetric or asymmetric B-CNN models in Keras with bilinear pooling, as proposed in [Bilinear CNNs for Fine-grained Visual Recognition](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf) (*ICCV*, 2015).

`bilinearpooling.pooling`:

- Average pooling of local feature vector outer products in `tensorflow`
- Includes element-wise signed square root and L2 normalization
- If using `combine`, you won't need to reference this explicitly

`bilinearpooling.combine`:

- Takes two `keras` models `fA` and `fB` with output shapes `(N, H, W, cA)`, `(N, H, W, cB)`
- Maps `[fA.output, fB.output]` to shape `(N, cA, cB)` with `bilinear.pooling`
- Flattens, connects to `softmax` output using a specifiable number of `Dense` layers.
- Returns the resulting `keras.models.Model` instance

#### Usage Notes

- Be careful with reuse of single model for `fA` and `fB` (*e.g.*, asymmetry via different output layers). Weights will be shared if you use the same instantiation of the original model to generate both models. This may or may not be desirable.

If the dimensionality of local feature vectors is 512, and there are `N` classes, the size of a fully-connected classification layer will be very large (`512*512*N=262,144*N`). With random weight initialization, it seems pretty difficult to train a layer of this size for moderate to large `N`.

## FV-CNN

The `texture.models.FVCNN` generates Fisher vector encodings from pretrained CNNs using the `cyvlfeat` wrappers for the `VLFeat` C library. A `FVCNN` instance can be constructed with an arbitrary CNN, or with a string specifying one of the supported ImageNet-pretrained models from `keras.applications`. A training set of images is required to generate the Gaussian Mixture Model of local feature vector distribution and train a support vector classifier. The training set can be a batch-style 4D numpy array, or a list of variable-size 3D image arrays.

## Benchmarks

Working on benchmarking models constructed with various texture recognition datasets. Some fine-grained classification datasets are also of interest, but benchmarking those has a lower priority for me at the moment.

- [Birds-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) (2011 version)
- [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## Further Improvements

- Add two-step training option (should be esp. useful for B-CNN nets).

#### Encoding

- Smaller `ResNet`-based constructors for feature networks

#### Bilinear

- Add support for `fA` and `fB` to have different input shapes (technically only output shapes need to correspond).
- Add support for `fA` and `fB` to have different output shapes (crop/interpolate/pool to match them)

Would also like to add the matrix square root normalization layer as described in:
```
@inproceedings{lin2017impbcnn,
    Author = {Tsung-Yu Lin, and Subhransu Maji},
    Booktitle = {British Machine Vision Conference (BMVC)},
    Title = {Improved Bilinear Pooling with CNNs},
    Year = {2017}}
```
Authors claim this improves accuracy by several % on fine-grained recognition benchmarks..
