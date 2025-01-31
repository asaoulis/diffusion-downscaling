"""Some tools for computing performance metrics over observations and predictions.

Here we combine the Metrics and SpatialAggregration classes to actually
compute the various defined metrics (mse, bias, crps) over multiple spatial
aggregation scales.
"""

import numpy as np
import torch
from functools import partial
import xarray as xr
from .metrics import Metrics
from .spatial_aggregation import SpatialAggregation

from .consts import HIST_LIMITS, DISTRIBUTION_BIAS_PARAMS


class Evaluation:
    """Take a range of metric configurations and compute the metrics.

    This information includes:
     - Multiple spatial aggregations - means
        and maxes over different spatial pooling scales
     - Which metrics to compute at each spatial aggregation level
     - Fractional skill score spatial scales and thresholds.
    """

    def __init__(self, spatial_aggregations={}, simple_metrics=[], fss_configs=[]):
        self.spatial_aggregations = spatial_aggregations
        self.metrics_to_run = simple_metrics
        self.fss_configs = fss_configs
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

    def compute_amax_stats(self, sample_xrs, eval_data):

        mean_precip_predictions = sample_xrs.mean(dim=("lon", "lat"))
        mean_precip_observations = eval_data.mean(dim=("lon", "lat"))
        years_dates = xr.date_range(
            mean_precip_predictions.time.values[0],
            mean_precip_predictions.time.values[-1],
            freq="Y",
        )
        amax_predictions = {}
        amax_top_observations = {}
        for years_date in years_dates:
            year = years_date.year

            yearly_data = mean_precip_observations.sel(
                time=mean_precip_observations.time.dt.year.isin([year])
            )
            top_precip_indices = np.argsort(yearly_data.values)[-5:][::-1]
            amax_precip_observations = yearly_data.isel(time=top_precip_indices)

            yearly_predictions = mean_precip_predictions.sel(
                time=mean_precip_predictions.time.dt.year.isin([year])
            )
            amax_precip_predictions = yearly_predictions.max(dim="time")

            amax_top_observations[year] = amax_precip_observations
            amax_predictions[year] = amax_precip_predictions.values

        return amax_predictions, amax_top_observations

    def compute_metrics(self, predictions, ground_truth, output_variable):
        """
        Evaluate the predictions.

        Returns:
        - metrics: Dictionary with evaluation metrics.
        """
        predictions = torch.Tensor(predictions).to(self.device)
        ground_truth = torch.Tensor(ground_truth).to(self.device)

        mean_preds = predictions.mean(dim=(0,1))
        eval_mean = ground_truth.mean(dim=(0))
        mean_error = mean_preds - eval_mean
        metrics = {}

        metrics["abs_bias"] = float(torch.abs(mean_error).mean())
        metrics["relative_bias"] = float((torch.abs(mean_error) / eval_mean).mean())

        dataset_views = self._build_data_views(predictions, ground_truth)
        metrics["pixelwise"] = {}
        for view, (predictions_view, ground_truth_view) in dataset_views.items():
            metrics["pixelwise"][view] = {}
            for metric in self.metrics_to_run:
                metrics["pixelwise"][view][metric] = Metrics.get_callable(metric)(
                    ground_truth_view, predictions_view
                )

        metrics["ralds"] = Metrics.radially_averaged_log_spectral_density(
            ground_truth, predictions
        )
        metrics["crps_ralds"] = Metrics.crps_radially_averaged_log_spectral_density(
            ground_truth, predictions
        )
        bin_range, transform = DISTRIBUTION_BIAS_PARAMS[output_variable]
        metrics["hists_crps"] = Metrics.crps_histograms(
            ground_truth, predictions, bin_range,
        )
        metrics['distribution_bias'] = float(Metrics.hist_distribution_bias(
            ground_truth, predictions, bin_range=bin_range, num_bins=25, transform=transform, return_all=False
        ))
        ground_truth = ground_truth.cpu().numpy()
        predictions = predictions.cpu().numpy()
        fss_metrics = self.compute_fss_metrics(predictions, ground_truth)
        metrics = {**metrics, **fss_metrics}

        thresholds = HIST_LIMITS[output_variable]
        metrics["calibration"] = self._compute_calibration_errors(
            predictions.transpose(1, 0, 2, 3), ground_truth, thresholds
        )
        metrics["event_ratios"] = Metrics.compute_event_ratios(
            predictions, ground_truth, thresholds
        )

        metrics["total_calibration"] = self._compute_calibration_errors(
            predictions.mean(axis=(-2,-1)).transpose(1, 0), ground_truth.mean(axis=(-2,-1)), 
            thresholds=[(-0.2, 10), (-0.1, 2), (2, 10), (5, 10)]
        )


        return metrics

    def compute_fss_metrics(self, predictions, ground_truth):
        fss_metrics = {}
        for fss_threshold, fss_scales in self.fss_configs:
            fss_metrics[f"fss_{fss_threshold}"] = []
            for scale in fss_scales:
                fss = Metrics.fraction_ensemble_skill_score(
                    ground_truth,
                    predictions,
                    threshold=fss_threshold,
                    window_radius=scale,
                )
                fss_metrics[f"fss_{fss_threshold}"].append(fss)
        return fss_metrics

    def _build_data_views(self, predictions, ground_truth):
        """Perform the various spatial aggregations and return the different views.
        """
        dataset_views = {"raw": (predictions, ground_truth)}
        for aggregation_type, factor in self.spatial_aggregations:
            aggregation_function = partial(
                SpatialAggregation.spatial_aggregation_to_coarse_grid,
                aggregation_type,
                factor,
            )
            dataset_views[f"{aggregation_type}_x{factor}"] = (
                aggregation_function(predictions),
                aggregation_function(ground_truth),
            )
        return dataset_views

    def _compute_calibration_errors(self, sample_arrays, observation_array, thresholds):
        precip_threshold_eces = {}
        for threshold in thresholds:

            hist = Metrics.compute_rank_histogram(
                observation_array, sample_arrays, threshold
            )
            precip_threshold_eces[f"ece_{threshold[0]}"] = (
                Metrics.compute_cdf_calibration_bias(hist)
            )

        return precip_threshold_eces
