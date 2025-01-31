
from .defaults import get_default_configs
from ..variables.all_var_predict_p import VARIABLES, VARIABLE_SCALER_MAP


def get_config():
    config = get_default_configs()
    # training
    config.base_output_dir = 'dir/diffusion-downscaling/scripts/'
    config.run_name = "determ_all_var_p"
    config.project_name = "multivariate_predictors"
    config.model_type = "deterministic"
    config.diffusion_type = "karras"
    config.precision = "bf16"

    training = config.training
    training.batch_size = 32
    training.n_epochs = 250
    training.loss_weights = None
    training.loss = "mse"

    # data
    data = config.data
    data.centered = True
    data.image_size = 128
    data.variables = VARIABLES
    data.variable_scaler_map = VARIABLE_SCALER_MAP
    data.dataset_path = "dataset.nc"
    data.train_indices = {"split": [0, 0.9]}
    data.eval_indices = {"split": [0.9, 1]}

    # model
    model = config.model
    model.conditional = False
    model.diffusion = False
    model.side_conditioning = False

    optim = config.optim
    optim.warmup = 0
    optim.beta1 = 0.9
    optim.lr = 1e-4
    optim.lr_schedule = {
        "step": {"step_size": 10, "gamma": 0.9}
    }  #'warmup': {'n_steps':500}

    return config
