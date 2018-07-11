**TODO**: 

- Tests and benchmarks
- `ResNet`-like (and other) network builders

# keras-texture

Implementations of several `keras` layers and other utilities that are useful in constructing models for texture recognition and fine-grained classification problems. It is a work in progress, and currently only the `tensorflow` backend is supported for the `keras` layers.

## `Encoding` Layer

The residual encoding layer proposed in [Deep TEN: Texture Encoding Network](https://arxiv.org/pdf/1612.02844.pdf) (*CVPR*, 2017). This `keras` implementation is based on the [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) release by the paper authors.

The layer learns a `KxD` codebook and codeword assignment `scale` weights used to encode an input of shape `NxD` or `HxWxD`. Includes optional L2 normalization of output vectors (`True` by default) and dropout (`None` by default). Unlike the `PyTorch` version, only `K` needs to be specified at construction time -- the feature size `D` is inferred from the `input_shape`.

## `BilinearModel` Layer

`bilinearmodel.BilinearModel` is a `keras` layer implementing the weighted outer product of two inputs with shape `[(batches,N),(batches,M)]`. This basic idea was first proposed in [Learning Bilinear Models for Two-Factor Problems in Vision](http://www.merl.com/publications/docs/TR96-37.pdf)(*CVPR*, 2017).

It is used in the `Deep Encoding Pooling Network (DEP)` proposed in [Deep Texture Manifold for Ground Terrain Recognition](https://arxiv.org/abs/1803.10896) (*CVPR*, 2018) to merge the output of an `Encoding` layer with the output of a standard global average pooling using the same `ResNet` feature extractor. The intuition is that the former represents textures (orderless encoding) and the latter represents spatially structured observations, so that "[the] outer product representation captures a pairwise correlation between the material texture encodings and spatial observation structures."

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

- Be careful with reuse of single model for `fA` and `fB` (*e.g.*, asymmetry via different output layers). Weights will be shared if you use the same instantiation of the original model to generate both models.

See `build_demo.ipynb` for examples of constructing symmetric and asymmetric B-CNNs using pretrained `VGG19` and `Xception` models from `keras.applications`.

#### Benchmarks

Working on benchmarking models constructed with this implementation on the three benchmark datasets referenced in the original B-CNN paper:

- [Birds-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) (2011 version)
- [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- [Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

You can run the `benchmark.py` script to build a model for `Birds-200`, after collecting dataset into `npy` files with `collect_dataset.py`:
```
$ python benchmark.py --help
```
Note: right now includes 1x1 conv to reduce `D` from `512 -> 32`. The former induces a fully connected layer w/ over 50 million weights (`200*512^2`), so training is very slow.

## Further Improvements

#### Encoding

- `ResNet` based constructors for feature networks

#### Bilinear `pooling`

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
Authors claim this improves accuracy by several % on fine-grained recognition benchmarks.

**Pull requests more than welcome!**
