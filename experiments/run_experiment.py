import argparse
import json
import importlib
from typing import Dict
import os

from util import train_model


DEFAULT_TRAIN_ARGS = {
    'batch_size': 32,
    'epochs': 1,
    'flags': []
}


def run_experiment(experiment_config: Dict, save: bool, gpu_ind: int):
    """
    experiment_config is of the form
    {
        "dataset": "DTDDataset",
        "dataset_args": {
            "data_dir": "../Dropbox/benchmark/dtd/",
            "input_size": 256
            "split": 7
        },
        "model": "TextureModel",
        "network": "deepten",
        "network_args": {
            "encode_K": 32,
            "dropout": 0.25
        },
        "train_args": {
            "batch_size": 16,
            "epochs": 50,
            "flags": ["TENSORBOARD", "CYCLIC_LR"]
            "flag_args": {"lr_low": 1e-4,
                          "lr_high": 1e-2}
        }
        "optimizer_args": {
            "optimizer": "SGD",
            "lr": 10e-3,
            "momentum": 0.9
        }
    }
    save_weights: if True, will save the final model weights to a canonical location (see TextureModel in models/base.py)
    gpu_ind: integer specifying which gpu to use

    If "LR_RAMP" or "CYCLIC_LR" in ["train_args"]["flags"], then a LearningRateScheduler callback will be created
    and SGD optimizer will be used. Otherwise, Adam will be used. In either case, optimizer_args should only
    contain valid keyword args for the given optimizer.
    """

    print(f'Running experiment with config {experiment_config} on GPU {gpu_ind}')

    datasets_module = importlib.import_module('texture.datasets')
    dataset_class_ = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module('texture.models')
    model_class_ = getattr(models_module, experiment_config['model'])

    networks_module = importlib.import_module('texture.networks')
    network_fn_ = getattr(networks_module, experiment_config['network'])
    network_args = experiment_config.get('network_args', {})

    optimizer_args = experiment_config.get('optimizer_args', {})

    model = model_class_(
        dataset_cls=dataset_class_,
        network_fn=network_fn_,
        dataset_args=dataset_args,
        network_args=network_args,
        optimizer_args=optimizer_args
    )
    print(model)

    experiment_config['train_args'] = {**DEFAULT_TRAIN_ARGS, **experiment_config.get('train_args', {})}
    experiment_config['experiment_group'] = experiment_config.get('experiment_group', None)
    experiment_config['gpu_ind'] = gpu_ind

    print("Training with flags: ", experiment_config['train_args']['flags'])

    train_model(
        model,
        dataset,
        epochs=experiment_config['train_args']['epochs'],
        batch_size=experiment_config['train_args']['batch_size'],
        flags=experiment_config['train_args']['flags'],
        flag_args=experiment_config['train_args'].get('flag_args', {}),
        gpu_ind=gpu_ind,
        save_ext=experiment_config.pop('save_ext', None)
    )
    score = model.evaluate(dataset.X_test, dataset.y_test)
    print(f'Test evaluation: {score}')


    if save:
        model.save_weights()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Provide index of GPU to use."
    )
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location"
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="JSON of experiment to run (e.g. '{\"dataset\": \"DTDDataset\", \"model\": \"TextureModel\", \"network\": \"bilinear_cnn\"}'"
    )
    args = parser.parse_args()


    experiment_config = json.loads(args.experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'
    run_experiment(experiment_config, args.save, args.gpu)
