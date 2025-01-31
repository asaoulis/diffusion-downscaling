spatial_aggregations = [
    ("mean", 4),
    ("mean", 16),
    ("mean", 64),
    ("ensemble", 1),
    ("ensemble", 16),
    ("ensemble", 64),
    ("max", 1),
    ("max", 4),
    ("max", 16),
    ("max", 64),
]
simple_metrics = ["mse", "crps", "bias", "absolute_bias", "relative_bias"]
fss_scales = [2, 4, 8, 16, 32, 64]
fss_configs = [(0.5, fss_scales), (1, fss_scales), (2, fss_scales), (5, fss_scales)]

EVAL_METRICS = (spatial_aggregations, simple_metrics, fss_configs)
