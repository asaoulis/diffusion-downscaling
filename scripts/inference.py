"""Script for running evaluation on a trained model with specified config file paths

This script is used to evaluate a trained model with specified config file paths. The script
loads the configuration files, builds the model, and runs evaluation on the model using the
specified sampling configuration.

Example:
    $ python inference.py -c training/configs/configs/gan.py -s training/configs/configs/gan.py

"""

from pathlib import Path
import argparse
import importlib

import diffusion_downscaling.lightning.utils as lightning_utils

from diffusion_downscaling.sampling.sampling import Sampler
from diffusion_downscaling.evaluation.utils import build_evaluation_callable
from diffusion_downscaling.sampling.utils import create_sampling_configurations
from diffusion_downscaling.data.constants import TRAINING_COORDS_LOOKUP

from configs.metrics.basic_eval_metrics import EVAL_METRICS as eval_metrics

import torch

torch.set_float32_matmul_precision("medium")


def parse_module(path):
    return path.replace(".py", "").replace("/", ".")


def run_eval(config, sampling_config, predictions_only):
    """Run evaluation over all sampling configurations give configs.

    This is a high level function that runs evaluation over all sampling configurations;
    it utilises the Sampler class to handle the evaluation process. The Sampler iterates
    over all requested sampling configurations, and runs the evaluation callable on each
    set of output predictions.

    :param config: The main configuration object, used to build the model and data scaler,
        as well as provide the output directory for the evaluation results.
    :param sampling_config: The sampling configuration object, used to specify sampling
        parameters as well as any extra data configuration settings (different datasets /
        indices to those used during training.)
    :param predictions_only: bool, flag indicating whether to generate predictions only.
    """

    config.data.eval_indices = sampling_config.eval_indices
    output_variables = config.data.variables[1]

    data_path = Path(config.data.dataset_path)
    config = lightning_utils.configure_location_args(config, data_path)
    location_config = dict(sampling_config.eval).get("location_config")

    base_output_dir = Path(config.base_output_dir)
    run_name = config.run_name
    project_name = config.project_name
    output_dir = base_output_dir / project_name / run_name
    data_scaler_path = sampling_config.get('data_scaler_path') or output_dir / 'scaler_parameters.pkl'
    data_scaler = lightning_utils.build_or_load_data_scaler(config, data_scaler_path)

    eval_config = sampling_config.eval
    checkpoint_name = eval_config.checkpoint_name
    model = lightning_utils.build_model(config, checkpoint_name)

    config.training.batch_size = sampling_config.batch_size
    custom_dset = dict(sampling_config).get("eval_dataset")
    config = lightning_utils.setup_custom_training_coords(config, sampling_config)

    if custom_dset is not None:
        data_path = custom_dset
        config.data.dataset_path = data_path
    if predictions_only:
        variables = (config.data.variables[0], [])
        data_scaler.set_transform_exclusion(config.data.variables[1])
        config.data.variables = variables    

    evaluation_sampler = Sampler(
        model,
        config.model_type,
        data_scaler,
        output_variables,
        sampling_config.output_format,
    )
    base_output = output_dir / eval_config.eval_output_dir
    num_samples = eval_config.n_samples

    eval_callable = build_evaluation_callable(
        data_path,
        sampling_config.eval_indices,
        eval_metrics,
        output_variables,
        predictions_only,
        config.residuals
    )

    if config.model_type == "diffusion":
        sampling_config.sampling.schedule.device = str(config.device)
        eval_args = create_sampling_configurations(
            sampling_config.sampling, location_config
        )
    elif config.model_type in ("gan", "deterministic"):
        eval_args = ({}, [({}, {}, {"location_config": location_config})])

    evaluation_sampler.evaluate_model_on_all_configs(
        config, eval_args, num_samples, base_output, eval_callable, output_variables
    )


def main(config_path, sampling_config_path, predictions_only):

    # Import config dynamically using provided config path
    config_module = importlib.import_module(parse_module(config_path))
    config = config_module.get_config()

    sampling_config_module = importlib.import_module(parse_module(sampling_config_path))
    sampling_config = sampling_config_module.get_sampling_config()

    print("Loaded configuration files.", flush=True)

    run_eval(config, sampling_config, predictions_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model with specified config file paths"
    )
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        default="training/configs/configs/gan.py",
        help="Path to the config file",
    )
    parser.add_argument(
        "-s",
        "--sampling-path",
        type=str,
        default="training/configs/configs/gan.py",
        help="Path to the sampling config file",
    )
    parser.add_argument(
        "-p",
        "--predictions-only",
        action="store_true",
        default=False,
        help="Flag indicating whether to generate predictions only",
    )
    args = parser.parse_args()

    main(args.config_path, args.sampling_path, args.predictions_only)
