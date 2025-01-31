
from .defaults import get_default_configs
from ..paths import TRAINING_DATA_PATH
from ..variables.all_var_predict_p import VARIABLES, VARIABLE_SCALER_MAP
import ml_collections


def get_config():
    config = get_default_configs()
    # training
    config.run_name = "patch_conv_soft_cl100_0.1gen"
    config.project_name = "gan_downscaling"
    config.model_type = "gan"
    config.precision = "bf16"

    training = config.training
    training.batch_size = 8
    training.n_epochs = 300
    training.loss_weights = [1.0]
    training.save_n_epochs = 5

    training.loss_config = ml_collections.ConfigDict()
    loss_config = training.loss_config
    loss_config.content_loss = True
    loss_config.content_batch = 4
    loss_config.lambda_content = 100.0
    loss_config.lambda_gp = 0

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
    data.location_config = 'colorado'


    # model
    model = config.model
    model.nonlinearity = "silu"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.embedding_type = "identity"
    model.diffusion = True

    config.discriminator = ml_collections.ConfigDict()
    config.discriminator = model.copy_and_resolve_references()
    discriminator = config.discriminator

    discriminator = config.discriminator
    discriminator.nf = 64
    discriminator.ch_mult = (1, 2, 2, 2)
    discriminator.num_res_blocks = 4
    discriminator.attn_resolutions = (8,)
    discriminator.conditional = False
    discriminator.nonlinearity = "lrelu"

    optim = config.optim
    optim.grad_clip = None
    optim.warmup = 0
    optim.beta1 = 0.9
    optim.lr = 5e-5
    optim.lr_schedule = {
        "step": {"step_size": 30, "gamma": 0.9}
    }  #'warmup': {'n_steps':500}}

    return config
