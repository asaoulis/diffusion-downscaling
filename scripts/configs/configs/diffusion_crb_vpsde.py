
from .defaults import get_default_configs
from ..paths import TRAINING_DATA_PATH
from ..variables.all_var_predict_p import VARIABLES, VARIABLE_SCALER_MAP

import ml_collections


def get_config():
    config = get_default_configs()
    # training
    config.run_name = "multipredict/vpsde_crb_p"
    config.project_name = "diffusion_downscaling"
    config.model_type = "diffusion"
    config.diffusion_type = "vpsde"
    config.precision = "bf16"

    config.model.side_conditioning = False
    config.model.cascade_conditioning = False
    config.model.location_parameters = None
    config.model.attention_type = "legacy"

    training = config.training
    training.batch_size = 24
    training.n_epochs = 250
    training.loss_weights = [1.0]
    training.loss_buffer_width = None


    # data
    data = config.data
    data.centered = True
    data.image_size = 128
    data.variables = VARIABLES
    data.variable_scaler_map = VARIABLE_SCALER_MAP
    data.dataset_path = TRAINING_DATA_PATH
    data.train_indices = {"split": [0, 0.9]}
    data.eval_indices = {"split": [0.9, 1]}
    data.variable_location = True
    data.variable_location_config = 'colorado'

    optim = config.optim
    optim.warmup = 0
    optim.beta1 = 0.9
    optim.lr = 2e-4
    optim.lr_schedule = {
        "step": {"step_size": 30, "gamma": 0.9}
    }  #'warmup': {'n_steps':500}}

    return config
