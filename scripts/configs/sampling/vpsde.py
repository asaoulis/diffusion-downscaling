import ml_collections


def get_sampling_config():

    config = ml_collections.ConfigDict()
    config.batch_size = 184
    config.output_format = "n_{n}_churn_{s_churn}_euler"
    config.eval_indices = {
        "split": [0.9, 1],
        "date_range": ("2012-01-01", "2013-12-31"),
    }

    config.eval = ml_collections.ConfigDict()
    eval = config.eval
    eval.eval_output_dir = "reduce_betamin"
    eval.location_config = ["colorado"]

    # eval.checkpoint_name = "epoch=144-val_loss=0.0066.ckpt"  #vpsde colorado
    eval.checkpoint_name = 'epoch=889-val_loss=0.0059.ckpt' #vpsde 
    eval.n_samples = 4

    config.sampling = ml_collections.ConfigDict()
    sampling = config.sampling

    sampling.sampler = ml_collections.ConfigDict()
    sampler = sampling.sampler
    sampler.integrator = "euler"
    sampler.s_churn = 3
    sampler.s_noise = 1.0
    sampler.s_tmin = 0.0001
    sampler.s_tmax = 1000
    sampler.sde_type = "vp"

    sampling.schedule = ml_collections.ConfigDict()
    schedule = sampling.schedule
    schedule.type = "vp"
    schedule.n = [39, 79]
    schedule.beta_d = 17.2
    schedule.beta_min = 0.1

    return config
