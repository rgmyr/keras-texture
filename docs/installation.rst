============
Installation
============

The ``keras-texture`` package is hosted on GitHub. You can clone the repo with::

    $ git clone https://github.com/rgmyr/keras-texture.git
        
The package can then be installed with the usual ``pip`` mechanisms::

    $ cd keras-texture
    $ pip install [-e] .

Requirements
============

- ``keras`` (version >=2.0) with ``tensorflow`` backend
- ``numpy``
- ``scikit-learn``
- ``scikit-image``

The _cyvlfeat wrappers are required for use the ``texture.fisher`` module. They should be installed with ``conda``, if at all possible.

.. _cyvlfeat: https://github.com/menpo/cyvlfeat

`TensorFlow` is not explicitly required, due to the ambiguity between the ``tensorflow`` and ``tensorflow-gpu`` packages. Either is technically acceptable as a ``keras.backend``. 

Most functionality in the package will require a GPU, but some models (*e.g.*, training an ``FV-CNN`` classifier on top of ImageNet pretrained models), can probably be used on a CPU-only machine.

