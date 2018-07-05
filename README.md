**Immediate TODO**: test training runs, make sure everything works as expected. Ideally, add scripts to benchmark performance against claims in original paper.

# bilinearCNN

Provides a few convenience functions for creating symmetric or asymmetric B-CNN models in Keras.

- `bilinear.pool`: Defines average pooling of feature-wise outer product in `tensorflow`, including element-wise signed square root and L2 normalization. If using `combine`, you won't need to reference this explicitly.
- `bilinear.combine`: Takes two `keras` models `fA` and `fB` with output shapes `(N, H, W, cA)`, `(N, H, W, cB)` and returns a single trainable model where `[fA.output, fB.output]` has been `bilinear.pool`ed and connected to a `softmax` output using a specifiable number of fully-connected layers.

*Note*: `bilinear.pool` does not include flattening, but `bilinear.combine` will add a `Flatten` layer before the `Dense` layer(s).

See `demo.ipynb` for examples of constructing symmetric and asymmetric B-CNNs using `keras.applications.vgg19`.

Original B-CNN paper:
```
@inproceedings{lin2015bilinear,
    Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
    Title = {Bilinear CNNs for Fine-grained Visual Recognition},
    Booktitle = {International Conference on Computer Vision (ICCV)},
    Year = {2015}}
```

## Further Improvements

Would like to add support for matrix square root normalization layer as described in:
```
@inproceedings{lin2017impbcnn,
    Author = {Tsung-Yu Lin, and Subhransu Maji},
    Booktitle = {British Machine Vision Conference (BMVC)},
    Title = {Improved Bilinear Pooling with CNNs},
    Year = {2017}}
```

Pull requests are more than welcomed.
