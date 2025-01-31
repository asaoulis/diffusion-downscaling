"""Various utilities to compute the relevant metrics we are interested in.

Includes:
    - Rank histograms and expected calibration errors
    - Radially averaged log spectral density
    - Various standard metrics such as mse, bias, crps
    - Event occurence ratios within different bins
"""

import numpy as np
import torch
import torch.nn.functional as F
from pysteps import utils
from pysteps.verification import probscores as ps
from pysteps.verification import ensscores as es
from scipy.signal import correlate


# pysteps rankhist with max values
def compute_rankhist(X_f, X_o, X_min=None, X_max=None, normalize=True):
    """Accumulate forecast-observation pairs to the given rank histogram.

    Parameters
    ----------
    rankhist: dict
      The rank histogram object.
    X_f: array-like
        Array of shape (k,m,n,...) containing the values from an ensemble
        forecast of k members with shape (m,n,...).
    X_o: array_like
        Array of shape (m,n,...) containing the observed values corresponding
        to the forecast.
    """
    X_f = X_f.copy()
    X_o = X_o.copy()
    num_ens_members = X_f.shape[0]
    rankhist = {}

    rankhist["num_ens_members"] = num_ens_members
    rankhist["n"] = np.zeros(num_ens_members + 1, dtype=int)
    rankhist["X_min"] = X_min
    if X_f.shape[0] != rankhist["num_ens_members"]:
        raise ValueError(
            "the number of ensemble members in X_f does not "
            + "match the number of members in the rank "
            + "histogram (%d!=%d)" % (X_f.shape[0], rankhist["num_ens_members"])
        )

    X_f = np.vstack([X_f[i, :].flatten() for i in range(X_f.shape[0])]).T
    X_o = X_o.flatten()

    X_min = rankhist["X_min"]

    mask = np.logical_and(np.isfinite(X_o), np.all(np.isfinite(X_f), axis=1))
    # ignore pairs where the verifying observations and all ensemble members
    # are below the threshold X_min
    if X_min is not None:
        mask_nz = np.logical_or(
            np.logical_and(X_o >= X_min, X_o <= X_max),
            np.logical_and(np.any(X_f >= X_min, axis=1), np.any(X_f <= X_max, axis=1)),
        )
        mask = np.logical_and(mask, mask_nz)

    X_f = X_f[mask, :].copy()
    X_o = X_o[mask].copy()
    if X_min is not None:
        X_f[X_f < X_min] = X_min - 1
        X_o[X_o < X_min] = X_min - 1

    X_o = np.reshape(X_o, (len(X_o), 1))

    X_c = np.hstack([X_f, X_o])
    X_c.sort(axis=1)

    idx1 = np.where(X_c == X_o)
    _, idx2, idx_counts = np.unique(idx1[0], return_index=True, return_counts=True)
    bin_idx_1 = idx1[1][idx2]

    bin_idx = list(bin_idx_1[np.where(idx_counts == 1)[0]])

    # handle ties, where the verifying observation lies between ensemble
    # members having the same value
    idxdup = np.where(idx_counts > 1)[0]
    if len(idxdup) > 0:
        X_c_ = np.fliplr(X_c)
        idx1 = np.where(X_c_ == X_o)
        _, idx2 = np.unique(idx1[0], return_index=True)
        bin_idx_2 = X_f.shape[1] - idx1[1][idx2]

        idxr = np.random.uniform(low=0.0, high=1.0, size=len(idxdup))
        idxr = bin_idx_1[idxdup] + idxr * (bin_idx_2[idxdup] + 1 - bin_idx_1[idxdup])
        bin_idx.extend(idxr.astype(int))

    for bi in bin_idx:
        rankhist["n"][bi] += 1

    if normalize:
        return 1.0 * rankhist["n"] / sum(rankhist["n"])
    else:
        return rankhist["n"]


# pysteps rapsd is slow
def rapds(data_array, fft_method=torch.fft, normalize=False, reduce=True):

    if torch.sum(torch.isnan(data_array)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = data_array.shape[-2:]

    yc, xc = utils.arrays.compute_centred_coord_array(m, n)

    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(data_array.shape[-2], data_array.shape[-1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(data_array))
        psd = torch.abs(psd) ** 2 / m * n
    else:
        psd = data_array
    result = []
    samples_shape = psd.shape
    if not reduce:
        psd = psd.reshape((-1, *samples_shape[2:]))
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd.detach().cpu().numpy()[np.broadcast_to(mask, psd.shape)]
        psd_vals = psd_vals.reshape(psd.shape[0], -1)
        psd_vals = np.mean(psd_vals, axis=1)
        if not reduce:
            psd_vals = psd_vals.reshape((*samples_shape[:2], -1))
        result.append(psd_vals)

    if reduce:
        result = np.array(result)
    else:
        result = np.array(result)
    if normalize:
        result /= np.sum(result)
    return result


class Metrics:

    @staticmethod
    def get_callable(metric):
        METRIC_TO_CALLABLE = {
            "mse": Metrics.mean_squared_error,
            "bias": Metrics.mean_error,
            "crps": Metrics.continuous_ranked_probability_score,
            'absolute_bias': Metrics.absolute_bias,
            'relative_bias': Metrics.relative_bias,
            "ralsd": Metrics.radially_averaged_log_spectral_density,
            "crps_ralsd": Metrics.crps_radially_averaged_log_spectral_density
            # 'correlation': Metrics.multivariate_correlation
        }
        return METRIC_TO_CALLABLE[metric]

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        # mse = F.mse_loss(torch.unsqueeze(y_true, dim=1), y_pred)
        if len(y_pred.shape) == 4:
            y_pred = y_pred.permute(1, 0, 2, 3)
        mse = F.mse_loss(y_true, y_pred)

        return mse.cpu().numpy()

    @staticmethod
    def mean_error(y_true, y_pred):
        if len(y_pred.shape) == 4:
            y_pred = y_pred.permute(1, 0, 2, 3)
        me = torch.mean(y_pred - y_true)
        return me.cpu().numpy()

    @staticmethod
    def absolute_bias(y_true, y_pred):
        if len(y_pred.shape) == 4:
            y_pred = y_pred.permute(1, 0, 2, 3)
        me = torch.mean(torch.abs(y_pred - y_true))
        return me.cpu().numpy()

    @staticmethod
    def relative_bias(y_true, y_pred):
        if len(y_pred.shape) == 4:
            y_pred = y_pred.permute(1, 0, 2, 3)
        me = torch.nanmean(torch.abs(y_pred - y_true) / y_true)
        return me.cpu().numpy()

    @staticmethod
    def continuous_ranked_probability_score(y_true, y_samples):
        """
        Calculate the Continuous Ranked Probability Score (CRPS).

        Parameters:
        - y_true: True values.
        - y_samples: Samples from the predictive distribution (e.g., probabilistic predictions).
        """
        try:
            if len(y_samples.shape) == 4:
                y_samples = y_samples.permute(1, 0, 2, 3)

            crps = ps.CRPS(
                y_samples.detach().cpu().numpy(), y_true.detach().cpu().numpy()
            )
        except ValueError:
            crps = np.NaN

        return crps

    # @staticmethod
    # def multivariate_correlation(y_true, y_samples):
    #     s_sh = y_samples.shape
    #     flat_shape = (s_sh[0] * s_sh[1], s_sh[])
    #     y_samples = y_samples.reshape(y_samples.shape[:2])

    @staticmethod
    def mass_conservation_violation(
        y_coarse, y_pred, coarse_grid_area, super_resolution_grid_area
    ):
        """
        Compute the mass conservation violation between coarse and super-resolution grids.

        Parameters:
        - coarse_values: Values on the coarse grid.
        - super_resolution_values: Values on the super-resolution grid.
        - coarse_grid_area: Area of each cell on the coarse grid.
        - super_resolution_grid_area: Area of each cell on the super-resolution grid.

        Returns:
        - mass_conservation_error: Absolute mass conservation violation.
        """
        coarse_total_mass = np.sum(y_coarse, axis=(1, 2)) * coarse_grid_area
        super_resolution_total_mass = (
            np.sum(y_pred, axis=(1, 2)) * super_resolution_grid_area
        )

        mass_conservation_error = np.mean(
            super_resolution_total_mass - coarse_total_mass
        )
        mass_conservation_bias = np.mean(
            super_resolution_total_mass - coarse_total_mass / coarse_total_mass
        )

        return mass_conservation_error, mass_conservation_bias

    @staticmethod
    def radially_averaged_log_spectral_density(
        y_true, y_samples, min_prec_threshold=0.02
    ):

        psd_true, psd_samples, _ = Metrics.compute_radially_averaged_spec_density(
            y_true, y_samples, min_prec_threshold
        )

        ralsd = np.sqrt(np.mean((10 * np.log10((psd_true) / psd_samples)) ** 2))

        return ralsd

    @staticmethod
    def crps_radially_averaged_log_spectral_density(
        y_true, y_samples, min_prec_threshold=0.02
    ):

        psd_true, psd_samples, _ = Metrics.compute_radially_averaged_spec_density(
            y_true, y_samples, min_prec_threshold, reduce=False
        )
        psd_true = psd_true.transpose(1, 0 )
        psd_samples = psd_samples.squeeze().transpose(2,1,0)
        true_ralsd = 10 * np.log10(psd_true)
        obs_ralsd = 10 * np.log10(psd_samples)
        score = Metrics.continuous_ranked_probability_score(torch.Tensor(true_ralsd), torch.Tensor(obs_ralsd))

        return score

    def hist_distribution_bias(
        y_true, y_samples, bin_range=(0.5, 250), num_bins=50, transform=torch.log10, return_all=False
    ):
        def prep_data(data):
            log_data = transform(data)
            log_data[torch.isnan(log_data) | (torch.isinf(log_data) & (log_data < 0))] = log_data.min() - 1
            return log_data

        log_eval = prep_data(y_true).flatten()
        log_samples = prep_data(y_samples).flatten()

        min_value, max_value = transform(torch.tensor(bin_range[0])), transform(torch.tensor(bin_range[1]))
        obs_hists = torch.histc(log_eval, bins=num_bins, min=min_value, max=max_value)
        obs_densities = torch.log10(obs_hists / obs_hists.sum())

        sample_hists = torch.histc(log_samples, bins=num_bins, min=min_value, max=max_value)
        sample_densities = torch.log10(sample_hists / sample_hists.sum())
        min_vals = [torch.where(obs_densities == -float('inf'), torch.tensor(float('inf')), obs_densities).min(), 
                    torch.where(sample_densities == -float('inf'), torch.tensor(float('inf')), sample_densities).min()]
        min_val = min(min_vals) - 0.5
        obs_densities = torch.where(obs_densities == -float('inf'), min_val, obs_densities)
        sample_densities = torch.where(sample_densities == -float('inf'), min_val, sample_densities)
        distribution_bias = (obs_densities - sample_densities).abs().mean()
        if return_all:
            return distribution_bias.cpu().numpy(), obs_densities, sample_densities
        return distribution_bias.cpu().numpy()
    def crps_histograms(
        y_true, y_samples, bin_range=(0.5, 250), num_bins=50, transform=torch.log10
    ):
        def prep_data(data):
            log_data = transform(data)
            log_data[torch.isnan(log_data) | (torch.isinf(log_data) & (log_data < 0))] = log_data.min() - 1
            return log_data

        log_eval = prep_data(y_true)
        log_samples = prep_data(y_samples)
        n, s, h, w = log_samples.shape
        flattened_samples = log_samples.reshape(-1, h, w)

        min_value, max_value = transform(torch.tensor(bin_range[0])), transform(torch.tensor(bin_range[1]))
        obs_hists = torch.stack([torch.histc(row, bins=num_bins, min=min_value, max=max_value) for row in log_eval])
        flat_sample_hists = torch.stack([torch.histc(row, bins=num_bins, min=min_value, max=max_value) for row in flattened_samples])
        flat_sample_hists = torch.log10(flat_sample_hists / flat_sample_hists.sum(dim=1, keepdim=True))

        sample_densities = flat_sample_hists.reshape(n, s, -1)
        # sample_densities = cleanup_logs(sample_densities)
        obs_densities = torch.log10(obs_hists / obs_hists.sum(dim=1, keepdim=True))
        # obs_densities = cleanup_logs(obs_densities)

        crps = Metrics.continuous_ranked_probability_score(obs_densities, sample_densities.permute(1,0,2))
        return crps
    @staticmethod
    def compute_radially_averaged_spec_density(
        y_true, y_samples, min_prec_threshold=0.02, reduce=True
    ):
        with torch.no_grad():
            indices = torch.where(torch.mean(y_true, dim=(1, 2)) > min_prec_threshold)[0].long().detach().cpu().numpy()
            y_true = y_true[indices]
            y_samples = y_samples[indices]

            psd_true = rapds(y_true)
            psd_samples = rapds(y_samples, reduce=reduce)

            _, freqs = utils.spectral.rapsd(
                y_true[0].detach().cpu().numpy(), return_freq=True
            )
            return psd_true, psd_samples, freqs

    @staticmethod
    def fraction_ensemble_skill_score(
        y_true, y_samples, threshold=0.5, window_radius=5
    ):
        """
        Compute the Fraction of Ensemble Skill Score (FSS).

        Parameters:
        - y_true: True values.
        - y_samples: Samples from the predictive distribution (e.g., probabilistic predictions).
        """
        fss_scores = np.vstack(
            [
                es.ensemble_skill(
                    samples, y_true[i], "fss", thr=threshold, scale=window_radius
                )
                for i, samples in enumerate(y_samples.transpose(1, 0, 2, 3))
            ]
        )

        return np.nanmean(fss_scores)

    def compute_rank_histogram(y_true, y_samples, thresholds):
        X_min, X_max = thresholds

        return compute_rankhist(y_samples, y_true, X_min, X_max, normalize=True)

    def compute_cdf_calibration_bias(hist):
        import scipy

        num_samples = len(hist)
        x = np.linspace(0, 1, num_samples)
        empirical_cdf = scipy.integrate.cumtrapz(hist, x * num_samples)
        empirical_cdf = np.concatenate([[0], empirical_cdf])
        return np.mean(empirical_cdf - x), np.mean((empirical_cdf - x) ** 2)

    def compute_event_ratios(sample_data, observations, thresholds):
        event_ratios = {}
        for threshold in thresholds:
            sample_density = (
                np.count_nonzero(
                    np.logical_and(
                        sample_data >= threshold[0], sample_data <= threshold[1]
                    )
                )
                / sample_data.size
            )
            obs_count = np.count_nonzero(
                np.logical_and(
                    observations >= threshold[0], observations <= threshold[1]
                )
            )
            observation_density = obs_count / observations.size
            ratio = (
                (sample_density / observation_density) if observation_density > 0 else 0
            )
            event_ratios[f"ratio_{threshold}"] = ratio, obs_count
        return event_ratios
