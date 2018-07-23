============
Installation
============

The project is hosted on GitHub. Get a copy by running::

    $ git clone https://github.com/rgmeyer/keras-texture.git
        
The package can then be installed with the usual `pip` mechanisms.

`TensorFlow` is not explicitly required, due to the ambiguity between the `tensorflow` and `tensorflow-gpu` packages. Either is acceptable as a `keras.backend`. Most functionality in the package will probably require a GPU, but some models (*e.g.*, an `FVCNN` with ImageNet pretrained models), could probably be used on a CPU-only machine.

