"""
Just here to define mandatory Dataset subclass methods.
"""

class Dataset:
    def data_dirname(self):
        raise NotImplementedError()

    def load_or_generate_data(self):
        raise NotImplementedError()
