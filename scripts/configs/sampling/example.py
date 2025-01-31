"""Annotated configuration file explaining some of the key configuration options.

Sampling files support running a product over lists of parameters.
For example, specifying:
    schedule.n = [10, 15, 20]
    schedule.rho = [3, 7, 15]
will run the sampling script 9 times, with the output for each configuration
saved in directories named according to config.output_format. 

The sampler, schedule, and location_config parameters all support this product.

"""
import ml_collections

def get_sampling_config():

    config = ml_collections.ConfigDict()
    # Batch size to use during inference - this will need to be reduced
    # to avoid OOM errors, especially if the domain is increased.
    config.batch_size = 32

    # Optional eval_dataset: if set, the original training dataset path
    # will not be used, and coarse_predictors will be loaded from the
    # specified file
    config.eval_dataset = "/data/climate/Downscaling/diffusion/merged/test.nc"

    # Static learnt fixed predictors can only be used if we also specify the
    # original training coordinates. 
    config.training_dataset = 'colorado'
    # Data scaler parameters path; avoids the need to recompute this, 
    # which would be undesirable as things can go wrong with moving parts
    config.data_scaler_path = "/data/climate/Downscaling/diffusion/data/scaling_parameters.pkl"

    # Output format for the saved samples. This will be used to name the
    # output directory for each separate sampling run, according to 
    # the named sampling parameters below.
    config.output_format = "{location_config}_n_{n}_rho_{rho}"

    # Specify the region of the dataset to run evaluation over.
    # date_range is optional; for the whole dataset, set 'split' = [0,1]
    config.eval_indices = {
        "split": [0.9, 1], "date_range": ("2012-01-01", "2013-12-31")
    }

    config.eval = ml_collections.ConfigDict()
    eval = config.eval
    # Regions to run evaluation over
    eval.location_config = ['colorado']#'full_patches' #'colorado'
    # Base output directory for the evaluation results
    eval.eval_output_dir = "historic"
    # Checkpoint path; can either be a full path, or just the checkpoint name
    # if checkpoint is in the default output path from the training run
    eval.checkpoint_name = "/data/climate/Downscaling/diffusion/checkpoints/epoch=1493-val_loss=0.0514.ckpt"

    # Number of samples to generate for sampling run
    eval.n_samples = 4

    config.sampling = ml_collections.ConfigDict()
    sampling = config.sampling

    # Integration technique and associated parameters.
    # Follows the EDM formulation - see code or technical report for details.
    sampling.sampler = ml_collections.ConfigDict()
    sampler = sampling.sampler
    # integration method; several options accepted (euler, heun)
    # but dpm2_heun proved by far best during testing
    sampler.integrator = "dpm2_heun"
    sampler.s_churn = 10
    sampler.s_noise = 1.005
    sampler.s_tmin = 0.04
    sampler.s_tmax = 50

    # Sampling schedule parameters - setting up the discrete steps
    # over noise levels to run the sampling integration over.
    sampling.schedule = ml_collections.ConfigDict()
    schedule = sampling.schedule
    # range of parameters accepted (vp, exponential, etc.)
    # but karras by far the best choice in testing
    schedule.type = "karras"
    # Key parameter: number of steps, n, to tradeoff accuracy vs. efficiency.
    # More steps is slower but generally glean better results.
    schedule.n = 20
    schedule.rho = 7
    schedule.sigma_min = 0.02
    schedule.sigma_max = 80

    return config
