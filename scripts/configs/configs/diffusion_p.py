
from .defaults import get_default_configs
from ..variables.all_var_p_diff import VARIABLES, VARIABLE_SCALER_MAP

import ml_collections


def get_config():
    config = get_default_configs()
    # training
    config.run_name = "multipredict/karras_p_diff_noscale"
    config.project_name = "diffusion_downscaling"
    config.model_type = "diffusion"
    config.diffusion_type = "karras"
    config.precision = "bf16"
    config.model.dropout = 0.1

    training = config.training
    training.batch_size = 32
    training.n_epochs = 250
    training.loss_weights = [1.0]

    # data
    data = config.data
    data.centered = True
    data.image_size = 128
    data.variables = VARIABLES
    data.variable_scaler_map = VARIABLE_SCALER_MAP
    data.dataset_path = (
        'dir/data/crb_diff_dataset.nc'
    )
    data.train_indices = {"split": [0, 0.9]}
    data.eval_indices = {"split": [0.9, 1]}

    optim = config.optim
    optim.warmup = 0
    optim.beta1 = 0.9
    optim.lr = 4e-4
    optim.lr_schedule = {
        "step": {"step_size": 100, "gamma": 0.9}
    }  #'warmup': {'n_steps':500}}

    return config
