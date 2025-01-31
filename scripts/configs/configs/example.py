"""Annotated configuration file explaining some of the key configuration options.
"""
import ml_collections
import torch

OUTPUTS_PATH = '/home/seb/Documents/Diffusion/diffusion_nstemp' # Output directory for logs, checkpoints, and later evaluation outputs
TRAINING_DATA_PATH = "/home/seb/Documents/test.nc" # Path to .nc file with coarse predictors and fine predictands
PROJECT_NAME = "diffusion_downscaling_with_temperature" # Project name
RUN_NAME = "test_100_epochs" # Run name
from ..variables.all_var_p import VARIABLES, VARIABLE_SCALER_MAP

from .defaults import get_default_configs

def get_config():
    config = get_default_configs()
    # Output directory for logs, checkpoints, and later evaluation outputs
    config.base_output_dir = OUTPUTS_PATH
    config.project_name = PROJECT_NAME
    config.run_name = RUN_NAME

    # Precision for training. Can be "32", "16" or "bf16".
    # "bf16" preferred but not always supported.
    config.precision = "bf16"
    config.model_type = "diffusion"
    config.diffusion_type = "karras"

    # GENERIC TRAINING OPTIONS 
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 16  # 128
    training.n_epochs = 100
    # Buffer around the output fields when calculating the loss. 
    # Experimental option to mitigate against boundary effects;
    # didn't improve performance
    training.loss_buffer_width = None
    training.loss_weights = [0.2, 1.0]

    # GENERIC DATA OPTIONS
    config.data = data = ml_collections.ConfigDict()
    data.image_size = 128
    # path to .nc file with coarse predictors and fine predictands
    data.dataset_path = 'path/to/data.nc'
    # path to .nc file with coarse predictors and fine predictands
    data.dataset_path = TRAINING_DATA_PATH
    # where are data.variables and data.variable_scaler_map defined?
    data.variables = VARIABLES
    data.variable_scaler_map = VARIABLE_SCALER_MAP
    # Fraction of the dataset to use as training and validation
    data.train_indices = {"split": [0, 0.9]}
    data.eval_indices = {"split": [0.9, 1]}
    # Train with randomly subselected patches
    data.variable_location = False
    # Alternatively, train only over a fixed region of the .nc file
    """ better remove and set as default? """
    data.location_config = None
    # Currently deprecated - could be used to train with date as conditional info
    data.time_inputs = False

    # MODEL ARCHITECTURE OPTIONS
    # Many of these are further explained in the code
    config.model = model = ml_collections.ConfigDict()
    model = config.model
    # Model architecture improvement, should set to True
    # See technical report documentation for details
    model.cascade_conditioning = False
    # Whether to include learnable location parameters in the model
    # Either None or an int for the number of channels.
    model.location_channels = 8 
    # Unused conditioning - used to be used for location info
    model.side_conditioning = False
    model.ema_rate = None
    # Attention calculation, options "legacy", "local", "rope"
    # "legacy" is standard, global attention; may scale badly to large input fields
    # "local" is local attention, "rope" is a rotary positional encoding
    model.attention_type = "legacy"
    # Number of features/channels - key parameter for varying model size
    model.nf = 128
    # Channel multipliers for each resolution block; 
    # used to increase depth of the UNet
    model.ch_mult = (1, 2, 2, 2)
    # Number of residual blocks at each level of the UNet
    model.num_res_blocks = 4
    # Rate at which to apply attention
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.dropout = 0.1

    # How to process the noise level (sigma) scalar conditioning information
    # Fourier best suited for our setup, but can be positional or
    model.embedding_type = "fourier"
    model.diffusion = True

    # Generic optimisation parameters - these are all safe choices
    # Varying may slightly improve convergence rate of model, but 
    # probably not too important
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000 # what is this? shouldn't it be 0?
    optim.grad_clip = 1.0

    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    return config
