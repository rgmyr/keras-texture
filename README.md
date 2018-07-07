**TODO**: Working on scripts to benchmark performance against claims in original paper.

# bilinearCNN

Provides a few convenience functions for creating symmetric or asymmetric B-CNN models in Keras, as proposed in:
```
@inproceedings{lin2015bilinear,
    Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
    Title = {Bilinear CNNs for Fine-grained Visual Recognition},
    Booktitle = {International Conference on Computer Vision (ICCV)},
    Year = {2015}}
```

#### Functions

`bilinear.pooling`:

- Defines average pooling of local feature vector outer products in `tensorflow`
- Includes element-wise signed square root and L2 normalization
- If using `combine`, you won't need to reference this explicitly

`bilinear.combine`: 

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

- Add support for `fA` and `fB` to have different input shapes (technically only output shapes need to correspond).
- Add support for `fA` and `fB` to have different output shapes (crop to match them)

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
