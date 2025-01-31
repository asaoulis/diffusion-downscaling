
from .defaults import get_default_configs
from ..variables.all_var_predict_pT_cesm import VARIABLES, VARIABLE_SCALER_MAP

import ml_collections


def get_config():
    config = get_default_configs()
    # training
    config.run_name = "multipredict/vpsde_sigma_in_lr2e-4"
    config.project_name = "diffusion_downscaling"
    config.model_type = "diffusion"
    config.diffusion_type = "vpsde"
    config.precision = "16"

    training = config.training
    training.batch_size = 32
    training.n_epochs = 250
    training.loss_weights = [1.0, 1.0]

    # data
    data = config.data
    data.centered = True
    data.image_size = 128
    data.variables = VARIABLES
    data.variable_scaler_map = VARIABLE_SCALER_MAP
    data.dataset_path = (
        "dir/data/pT_elevation_1985_2015_dataset.nc"
    )
    data.train_indices = {"split": [0, 0.9]}
    data.eval_indices = {"split": [0.9, 1]}

    optim = config.optim
    optim.warmup = 0
    optim.beta1 = 0.9
    optim.lr = 2e-4
    optim.lr_schedule = {
        "step": {"step_size": 50, "gamma": 0.9}
    }  #'warmup': {'n_steps':500}}

    return config
