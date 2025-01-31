import ml_collections


def get_sampling_config():

    config = ml_collections.ConfigDict()
    config.batch_size = 64
    config.output_format = "results"
    config.eval_dataset = "eval.nc"

    config.eval_indices = {
        "split": [0, 1],
        # "date_range": ("2012-01-01", "2013-12-31"),
        "date_range": ("2015-01-01", "2018-12-31"),
    }

    config.eval = ml_collections.ConfigDict()
    eval = config.eval
    eval.eval_output_dir = "final"
    eval.checkpoint_name = "epoch=56-val_loss=0.0815.ckpt"
    eval.n_samples = 0
    eval.location_config = "colorado"

    return config
