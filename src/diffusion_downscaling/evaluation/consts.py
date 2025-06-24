"""Some presets for computing our histograms and weighted probabilistic
metrics for the various output variables. These are basically just bins
over which we plot some metrics and compute average performance over.
"""
import numpy as np
import torch

HIST_LIMITS = {
    "precipitation": [
        (0, 1),
        (1, 10),
        (10, 25),
        (25, 50),
        (50, 200),
    ],
    "air_temperature": [
        (-np.inf, -20),
        (-20, -10),
        (-10, 0),
        (0, 10),
        (10, 20),
        (20, 30),
        (30, np.inf),
    ],
}

DISTRIBUTION_BIAS_PARAMS = {
    "precipitation": ((0.5, 200), torch.log10),
    "air_temperature": ((-30, 42), lambda x: x)
    }