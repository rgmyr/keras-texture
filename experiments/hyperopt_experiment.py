import os
import argparse
import json
import importlib

import numpy as np
from scipy.optimize import curve_fit

import hyperopt
import hyperopt_config
from train_model import train_model


# results saved to join(DEFAULT_SAVE_LOC, args.name)
DEFAULT_SAVE_LOC = '/home/'+os.environ['USER']+'/Dropbox/benchmark/saved_models'

DEFAULT_TRAIN_ARGS = {
    'batch_size': 8,
    'epochs': 25,
    'flags': ['TENSORBOARD']
}

def val_loss_fit_func(x, a, b, c):
    '''Function that gets fit for loss estimate optimization.'''
    return a * np.exp(-b * x) + c

def make_train_function(experiment_config, save_dir_function):
    '''
    `experiment_config` is json dict of the form
    {
        "dataset": "DTDDataset",
        "dataset_args": {
            "input_shape": (256,256,3),
            "seed": 27
        },
        "model" : "NetworkModel",
        "model_args" : {
            "network" : "deepten"
            "network_args" : "hyperopt"
        }
        "predictor_model": "XGB"
        "predictor_model_args": {
            network
        }
    }
    Any `train_args` will override values in DEFAULT_TRAIN_ARGS values.
    Search spaces for `network_args` and `optimizer_args` are defined in `tune_config.py`
    `save_dir_function` makes unique names from tunable parameter values.
    '''
    datasets_module = importlib.import_module('texture.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module('texture.models')
    model_class_ = getattr(models_module, experiment_config['model'])
    print("Setup model_class...")

    networks_module = importlib.import_module('texture.networks')
    network_fn_ = getattr(networks_module, experiment_config['network'])
    print("Setup network_fn...")

    train_args = {**DEFAULT_TRAIN_ARGS, **experiment_config.get('train_args', {})}

    def train_function(config):
        model = model_class_(
            dataset_cls=dataset_class_,
            #network_fn=network_fn_,
            dataset_args=dataset_args,
            model_args=config,
            network_args=config['network_args'],
            optimizer_args=config['optimizer_args']
        )
        train_model(
            model,
            dataset,
            epochs=train_args['epochs'],
            batch_size=train_args['batch_size'],
            flags=train_args.get('flags', {}),
            save_dir=save_dir_function(config)
        )
        # fit the validation loss, optimize w.r.t. val_loss at epoch=(5*epochs)
        val_loss = model.val_loss
        num_epochs = train_args['epochs']

        popt, pcov = curve_fit(val_loss_fit_func, np.arange(num_epochs), val_loss)
        estimated_loss = val_loss_fit_func(5*num_epochs, *popt)
        print("Estimated val_loss at epoch", num_epochs*5, ": ", estimated_loss)

        return estimated_loss

    return train_function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default='unnamed_experiment',
        help="Name of experiment (creates a directory in DEFAULT_SAVE_LOC)."
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=100,
        help="Number of max_evals for hyperopt.fmin(), default=100."
    )
    parser.add_argument(
        "--tpe",
        action="store_true",
        help="If present, use Tree of Parzen Estimators instead of Random search algorithm."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA index of GPU to run experiments on, default=0."
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="JSON of experiment to run (see make_train_function() docstring).'"
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'

    experiment_config = json.loads(args.experiment_config)
    search_space = hyperopt_config.search_spaces[experiment_config['network']]

    make_unique_name = hyperopt_config.make_unique_name_function(search_space)
    save_dir_function = lambda params: os.path.join(DEFAULT_SAVE_LOC, args.name, make_unique_name(params))

    best_params = hyperopt.fmin(
        fn=make_train_function(experiment_config, save_dir_function),
        space=search_space,
        algo=hyperopt.tpe.suggest if args.tpe else hyperopt.random.suggest,
        max_evals=args.max_evals
    )
    print("Best hyperparameters found:\n", best_params)
