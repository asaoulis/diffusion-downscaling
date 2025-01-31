"""Wide range of plotting functions for model benchmarking.
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1


class EvaluationPlotter:

    plotting_presets = {
        "precipitation": {"levels": 100, "vmin": 0, "vmax": 30, "cmap": "Blues"},
        "air_temperature": {"levels": 100, "vmin": -30, "vmax": 30, "cmap": "RdBu_r"},
    }

    units_presets = {"precipitation": "mm/day", "air_temperature": "C"}

    def __init__(self, output_dir, output_variable):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_variable = output_variable
        self.contourf_kwargs = self.plotting_presets[output_variable]
        self.units = self.units_presets[output_variable]

        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="110m",
            facecolor="none",
        )
        self.features = [states_provinces, cfeature.RIVERS, cfeature.BORDERS]

    def _add_geographic_features(self, ax, features):
        ax.coastlines()
        for feature in features:
            ax.add_feature(feature)

    def plot_fss(self, fss_scores, scales, threshold):
        plt.plot(scales, fss_scores)
        plt.xlabel("Scale (km)")
        plt.ylabel("FSS")
        plt.title(f"Fractional skill score, threshold {threshold} {self.units}")
        plt.savefig(self.output_dir / f"fss_{threshold}.png")
        plt.close()

    def plot_all_fss(self, all_fss_scores, scales):
        plt.figure(figsize=(10, 10))
        for threshold, scores in all_fss_scores.items():
            plt.plot(scales, scores, label=f"{threshold} ({self.units})")
        plt.legend()
        plt.xlabel("Scale (km)")
        plt.ylabel("FSS")
        plt.title("Fractional skill scores @ different thresholds")
        plt.savefig(self.output_dir / f"fss_varying_thresholds.png")
        plt.close()

    def plot_mean_spatial_bias(self, member_mean_preds, eval_data):
        mean_preds = member_mean_preds[self.output_variable].mean(dim="member")
        eval_mean = eval_data[self.output_variable].mean(dim="time")
        mean_error = mean_preds - eval_mean
        relative_error = mean_error / eval_mean * 100
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 2, figsize=(20, 20), subplot_kw=dict(projection=ccrs.PlateCarree())
        )
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="110m",
            facecolor="none",
        )

        eval_mean.plot.contourf(
            levels=100,
            cmap=self.contourf_kwargs["cmap"],
            vmin=eval_mean.min(),
            vmax=eval_mean.max(),
            ax=axes[0, 0],
            transform=ccrs.PlateCarree(),
        )
        axes[0, 0].set_title(f"Daily mean observation ({self.units})")

        mean_preds.plot.contourf(
            levels=100,
            cmap=self.contourf_kwargs["cmap"],
            ax=axes[0, 1],
            vmin=eval_mean.min(),
            vmax=eval_mean.max(),
            transform=ccrs.PlateCarree(),
        )
        axes[0, 1].set_title(f"Daily mean prediction ({self.units})")
        mean_error.plot.contourf(
            levels=100, cmap="RdBu", ax=axes[1, 0], transform=ccrs.PlateCarree()
        )
        axes[1, 0].set_title(f"Daily mean bias ({self.units})")
        relative_error.plot.contourf(
            levels=100,
            cmap="RdBu",
            ax=axes[1, 1],
            vmax=100,
            transform=ccrs.PlateCarree(),
        )
        axes[1, 1].set_title("Daily relative bias (%)")
        for ax in axes.ravel():
            self._add_geographic_features(ax, self.features)
        plt.savefig(self.output_dir / "mean_spatial_bias.png")
        plt.close()

    def plot_amax_spatial_bias(self, xr_samples, eval_data, agg_type="mean"):
        if agg_type == "mean":
            max_preds = (
                xr_samples[self.output_variable].max(dim=("time")).mean(dim="member")
            )
        elif agg_type == "max":
            max_preds = (
                xr_samples[self.output_variable].max(dim=("time")).max(dim="member")
            )
        eval_max = eval_data[self.output_variable].max(dim="time")
        mean_error = max_preds - eval_max
        relative_error = mean_error / eval_max * 100
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 2, figsize=(20, 20), subplot_kw=dict(projection=ccrs.PlateCarree())
        )
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="110m",
            facecolor="none",
        )
        vmin = min(eval_max.min(), max_preds.min())
        vmax = min(eval_max.max(), max_preds.max())
        eval_max.plot.contourf(
            levels=100,
            cmap=self.contourf_kwargs["cmap"],
            ax=axes[0, 0],
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        axes[0, 0].set_title(f"Yearly amax observation ({self.units})")

        max_preds.plot.contourf(
            levels=100,
            cmap=self.contourf_kwargs["cmap"],
            ax=axes[0, 1],
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        axes[0, 1].set_title(f"Yearly amax prediction ({self.units})")
        mean_error.plot.contourf(
            levels=100, cmap="RdBu", ax=axes[1, 0], transform=ccrs.PlateCarree()
        )
        axes[1, 0].set_title(f"Yearly amax bias ({self.units})")
        relative_error.plot.contourf(
            levels=100,
            cmap="RdBu",
            ax=axes[1, 1],
            vmax=100,
            transform=ccrs.PlateCarree(),
        )
        axes[1, 1].set_title("Yearly amax relative bias (%)")
        for ax in axes.ravel():
            self._add_geographic_features(ax, self.features)
        plt.savefig(self.output_dir / f"amax_{agg_type}_spatial_bias.png")
        plt.close()

    def plot_amax_spatial_sigmas(self, xr_samples, eval_data):
        amaxs = xr_samples[self.output_variable].max(dim=("time"))
        max_preds = amaxs.mean(dim="member")
        std_preds = amaxs.std(dim="member")

        eval_max = eval_data[self.output_variable].max(dim="time")
        mean_error = max_preds - eval_max
        relative_error = mean_error / std_preds
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 2, figsize=(20, 20), subplot_kw=dict(projection=ccrs.PlateCarree())
        )
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="110m",
            facecolor="none",
        )
        vmin = min(eval_max.min(), max_preds.min())
        vmax = min(eval_max.max(), max_preds.max())
        eval_max.plot.contourf(
            levels=100,
            cmap=self.contourf_kwargs["cmap"],
            ax=axes[0, 0],
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        axes[0, 0].set_title(f"Yearly amax observation ({self.units})")

        max_preds.plot.contourf(
            levels=100,
            cmap=self.contourf_kwargs["cmap"],
            ax=axes[0, 1],
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        axes[0, 1].set_title(f"Yearly amax prediction ({self.units})")
        mean_error.plot.contourf(
            levels=100, cmap="RdBu", ax=axes[1, 0], transform=ccrs.PlateCarree()
        )
        axes[1, 0].set_title(f"Yearly amax absolute bias ({self.units})")
        relative_error.plot.contourf(
            levels=100, cmap="RdBu", ax=axes[1, 1], vmax=5, transform=ccrs.PlateCarree()
        )
        axes[1, 1].set_title("Yearly amax sigma bias ($\sigma$)")
        for ax in axes.ravel():
            self._add_geographic_features(ax, self.features)
        plt.savefig(self.output_dir / f"amax_sigma_spatial_bias.png")
        plt.close()

    def plot_power_spectra(self, power_spectra, wavelengths):
        for name, spectrum in power_spectra.items():
            plt.semilogx(wavelengths, spectrum, label=name)
        plt.xlabel("Wavelength (km)")
        plt.ylabel("Power density")
        plt.title("Radially averaged spectral densities")
        plt.legend()
        plt.savefig(self.output_dir / "ralsd_spectra.png")
        plt.close()

    def plot_samples(
        self,
        sample_xrs,
        eval_data,
        num_samples=5,
        num_plots=30,
        vmin=0,
        vmax=30,
        levels=100,
        cmap="Blues",
        **plotting_kwargs,
    ):
        import math as m

        sample_xrs = sample_xrs[self.output_variable]
        num_samples = min(num_samples, len(sample_xrs.member))
        num_samples = max(3, num_samples)
        figure_square_size = m.ceil(m.sqrt(num_samples))
        output_dir = self.output_dir / "samples"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_plots):
            fig, axs = plt.subplots(
                figure_square_size,
                2 * figure_square_size,
                layout="constrained",
                figsize=(8 * figure_square_size, 4 * figure_square_size),
            )
            gridspec = axs[0, 0].get_subplotspec().get_gridspec()
            for a in axs.ravel():
                a.remove()

            ground_truth_fig = fig.add_subfigure(gridspec[:, :figure_square_size])
            axs_ground_truth = ground_truth_fig.subplots(
                3, 1, gridspec_kw=dict(height_ratios=[1, 3, 1], width_ratios=[1])
            )
            training_data = eval_data.isel(time=i)
            for ax in [axs_ground_truth[0], axs_ground_truth[-1]]:
                ax.remove()
            training_data[self.output_variable].plot.contourf(
                ax=axs_ground_truth[1], **self.contourf_kwargs
            )
            axs_ground_truth[1].set_title("Ground truth MSWEP")

            samples_fig = fig.add_subfigure(gridspec[:, figure_square_size:])
            samples_fig.suptitle("Samples")
            axs_ground_truth = samples_fig.subplots(
                figure_square_size, figure_square_size
            )
            for j, ax in zip(range(num_samples), axs_ground_truth.ravel()):
                sample_xrs.isel(time=i, member=j).plot.contourf(
                    ax=ax, add_colorbar=False, **self.contourf_kwargs
                )
                ax.set_axis_off()
                ax.set_title("")

            gridspec.tight_layout(samples_fig)
            plt.savefig(
                output_dir
                / f"{np.mean(training_data[self.output_variable].values):.4f}_prec.png"
            )
            plt.close()

    def plot_timelapse(
        self,
        sample_xrs,
        eval_data,
        num_plots=5,
        num_days=5,
        num_samples=4,
        vmin=0,
        vmax=30,
        levels=100,
        cmap="Blues",
        **plotting_kwargs,
    ):
        import xarray as xr

        sample_xrs = sample_xrs[self.output_variable]
        eval_data = eval_data[self.output_variable]
        output_dir = self.output_dir / "time_lapses"
        output_dir.mkdir(parents=True, exist_ok=True)
        num_days = int(round_up_to_odd(num_days))
        offset = int(np.ceil(num_days) // 2)

        mean_precip_observations = eval_data.mean(dim=("lon", "lat"))

        top_precip_indices = np.argsort(mean_precip_observations.values)[::-1][
            :num_plots
        ]

        for i in range(num_plots):
            data_index_isel = top_precip_indices[i]
            fig, axs = plt.subplots(
                num_samples + 1, num_days, figsize=(5 * num_days, 4 * num_samples)
            )
            for row_index in range(num_samples + 1):
                row_ax = axs[row_index, :]
                for j, ax in enumerate(row_ax.ravel()):
                    current_data_idx = -offset + j + data_index_isel
                    if row_index == 0:
                        try:
                            day_observation = eval_data.isel(time=current_data_idx)
                        except IndexError:
                            day_observation = xr.zeros_like(
                                eval_data.isel(time=data_index_isel)
                            )

                        day_observation.plot.contourf(
                            ax=ax, add_colorbar=False, **self.contourf_kwargs
                        )
                        if j == offset:
                            mean_precip = mean_precip_observations.isel(
                                time=data_index_isel
                            )
                            date = np.datetime_as_string(
                                mean_precip.time.values[()], unit="D"
                            )
                            ax.set_title(
                                f"{date}, {mean_precip.values:.3f} ({self.units})"
                            )
                        else:
                            ax.set_title(f"{-offset + j} days offset")
                    else:
                        try:
                            sample_prediction = sample_xrs.isel(
                                time=current_data_idx, member=row_index - 1
                            )
                        except IndexError:
                            sample_prediction = xr.zeros_like(
                                sample_xrs.isel(
                                    time=data_index_isel, member=row_index - 1
                                )
                            )

                        sample_prediction.plot.contourf(
                            add_colorbar=False, ax=ax, **self.contourf_kwargs
                        )
                        ax.set_axis_off()
                        ax.set_title("")
            plt.savefig(
                output_dir
                / f"{date}, {np.mean(mean_precip.values):.4f}_{self.output_variable}.png",
                dpi=200,
            )
            plt.close()

    def plot_multivariate_timelapse(
        self,
        sample_xrs_all_vars,
        eval_data_all_vars,
        variables,
        num_plots=5,
        num_days=5,
        num_samples=4,
        vmin=0,
        vmax=30,
        levels=100,
        cmap="Blues",
        **plotting_kwargs,
    ):
        import xarray as xr

        output_dir = self.output_dir / "multivariate_time_lapses"
        output_dir.mkdir(parents=True, exist_ok=True)
        num_days = int(round_up_to_odd(num_days))
        offset = int(np.ceil(num_days) // 2)

        mean_precip_observations = eval_data_all_vars.precipitation.mean(
            dim=("lon", "lat")
        )

        top_precip_indices = np.argsort(mean_precip_observations.values)[::-1][
            :num_plots
        ]

        for i in range(num_plots):
            data_index_isel = top_precip_indices[i]
            fig, axs = plt.subplots(
                (num_samples + 1) * len(variables),
                num_days,
                figsize=(5 * num_days, 4 * (num_samples + 1) * len(variables)),
            )
            for var_idx, variable in enumerate(variables):
                row_multiplier = lambda row_idx: row_idx * len(variables) + var_idx
                sample_xrs = sample_xrs_all_vars[variable]
                eval_data = eval_data_all_vars[variable]
                contourf_kwargs = self.plotting_presets[variable]

                for loop_row_index in range(num_samples + 1):
                    ax_row_index = row_multiplier(loop_row_index)
                    row_ax = axs[ax_row_index, :]
                    for j, ax in enumerate(row_ax.ravel()):
                        current_data_idx = -offset + j + data_index_isel
                        if loop_row_index == 0:
                            try:
                                day_observation = eval_data.isel(time=current_data_idx)
                            except IndexError:
                                day_observation = xr.zeros_like(
                                    eval_data.isel(time=data_index_isel)
                                )

                            day_observation.plot.contourf(
                                ax=ax, add_colorbar=False, **contourf_kwargs
                            )
                            if ax_row_index == 0:
                                if j == offset:
                                    mean_precip = mean_precip_observations.isel(
                                        time=data_index_isel
                                    )
                                    date = np.datetime_as_string(
                                        mean_precip.time.values[()], unit="D"
                                    )
                                    ax.set_title(
                                        f"{date}, {mean_precip.values:.3f} ({self.units})"
                                    )
                                else:
                                    ax.set_title(f"{-offset + j} days offset")
                            else:
                                ax.set_axis_off()
                                ax.set_title("")
                        else:
                            try:
                                sample_prediction = sample_xrs.isel(
                                    time=current_data_idx, member=loop_row_index - 1
                                )
                            except IndexError:
                                sample_prediction = xr.zeros_like(
                                    sample_xrs.isel(
                                        time=data_index_isel, member=loop_row_index - 1
                                    )
                                )

                            sample_prediction.plot.contourf(
                                add_colorbar=False, ax=ax, **contourf_kwargs
                            )
                            ax.set_axis_off()
                            ax.set_title("")
            plt.savefig(
                output_dir / f"{date}, {np.mean(mean_precip.values):.4f}_precip.png",
                dpi=200,
            )
            plt.close()

    def scatter_plot(self, samples_scatter, observations_scatter):

        plt.figure(figsize=(10, 8))  # Larger figure size

        # Using gridspec_kw to control the aspect ratios and size of the subplots
        gs = plt.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

        # Scatter plot with line of best fit
        ax1 = plt.subplot(gs[0])
        ax1.scatter(observations_scatter, samples_scatter)
        m, b = np.polyfit(observations_scatter, samples_scatter, 1)
        # Calculate R-squared
        y_pred = m * observations_scatter + b
        residuals = samples_scatter - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((samples_scatter - np.mean(samples_scatter)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        x = np.linspace(min(observations_scatter), max(observations_scatter), 100)
        ax1.plot(x, m * x + b, color="red", label=f"Best fit, $R^2$ = {r_squared:.2f}")
        ax1.plot(x, x, color="black", linestyle="--", label="Ideal")
        ax1.set_ylabel(f"Sample {self.output_variable}")
        ax1.legend()

        if np.max(samples_scatter) > np.max(observations_scatter):
            hist, bins = np.histogram(samples_scatter, bins=20)
        else:
            hist, bins = np.histogram(observations_scatter, bins=20)
        ax2 = plt.subplot(gs[2])
        ax2.hist(observations_scatter, bins=bins, color="blue", alpha=0.5, density=True)
        ax2.set_xlabel(f"Observation {self.output_variable}")

        # Histogram for y to the side of the scatter plot
        ax3 = plt.subplot(gs[1])
        ax3.hist(
            samples_scatter,
            bins=bins,
            orientation="horizontal",
            color="green",
            alpha=0.5,
            density=True,
        )

        plt.tight_layout()
        plt.savefig(self.output_dir / f"mean_{self.output_variable}_scatter.png")
        plt.close()

    def plot_rank_histogram(self, hist, name):

        num_samples = len(hist)
        if num_samples < 2 or not np.isfinite(np.max(hist)):
            return
        x = np.linspace(0, 1, num_samples)
        expected = np.ones_like(x) * 1 / num_samples
        plt.plot(x, hist, label="Observed distribution")
        plt.plot(x, expected, color="black", linestyle="--", label="Ideal")
        plt.ylim([0, np.max(hist) + 0.1 / num_samples])
        plt.fill_between(x, hist, expected, color="r", alpha=0.5)
        plt.legend()
        plt.title(f"Rank histogram over all {name}")
        plt.xlabel("Normalized Rank")
        plt.ylabel("Normalized Occurence")
        plt.savefig(self.output_dir / f"{name}_rank_histogram.png")
        plt.close()

    def plot_empirical_cdf(self, hist, name):
        import scipy

        num_samples = len(hist)
        x = np.linspace(0, 1, num_samples)
        empirical_cdf = scipy.integrate.cumtrapz(hist, x * num_samples)
        empirical_cdf = np.concatenate([[0], empirical_cdf])
        plt.plot(x, empirical_cdf, label="Observed distribution")
        plt.plot(x, x, color="black", linestyle="--", label="Ideal")
        plt.fill_between(x, empirical_cdf, x, color="r", alpha=0.5)
        plt.legend()
        plt.title(f"Empirical CDF over all {name}")
        plt.xlabel("Normalized Rank")
        plt.ylabel("Empirical CDF")
        plt.savefig(self.output_dir / f"{name}_empirical_CDF.png")
        plt.close()

    def plot_histogram_comparison(
        self, mean_samples, mean_observations, comparison_name, logx=True
    ):
        combined = np.concatenate(
            [mean_observations, mean_samples]
        )  # .clip(0.1, max=None)
        if logx:
            combined = combined.clip(0.1, max=None)
        min_value = np.min(combined)
        max_value = np.max(combined)
        num_bins = 50

        # if np.max(mean_samples) < np.max(mean_observations):
        #     hist, bins = np.histogram(mean_samples, bins=50)
        # else:
        #     hist, bins = np.histogram(mean_observations, bins=50)
        if logx:
            logbins = np.logspace(np.log10(min_value), np.log10(max_value), num_bins)
        else:
            logbins = np.linspace(min_value, max_value, num_bins)
        plt.figure(figsize=(10, 5))
        plt.title(f"{comparison_name} distributions")
        pred_densities, _, _ = plt.hist(
            mean_samples, bins=logbins, label="samples", density=True, histtype="step"
        )
        obs_densities, _, _ = plt.hist(
            mean_observations,
            bins=logbins,
            label="observed",
            density=True,
            histtype="step",
        )
        combined = np.concatenate([pred_densities, obs_densities])
        min_non_zero_density = np.min(combined[np.nonzero(combined)])
        if logx:
            plt.xscale("log")

        plt.yscale("symlog", linthresh=min_non_zero_density)
        plt.ylim([min_non_zero_density, 100])
        if logx:
            plt.xlim([0.1, max_value])

        plt.legend()

        plt.xlabel(f"{comparison_name} ({self.units})")
        plt.ylabel("Normalized frequency")
        plt.savefig(self.output_dir / f"{comparison_name}_distribution.png")
        plt.close()

    def plot_annual_maximum_distributions(
        self, amax_predictions, amax_top_observations
    ):
        num_years = len(amax_top_observations)
        fig, axs = plt.subplots(num_years, 1, figsize=(10, 3 * num_years), sharex=True)
        max_global_precip = np.max(
            [
                max(max(amax_samples), max(top_observations))
                for amax_samples, top_observations in zip(
                    amax_predictions.values(), amax_top_observations.values()
                )
            ]
        )
        if num_years == 1:
            axs = np.array([axs])
        for ax, year in zip(axs.ravel(), amax_top_observations.keys()):
            top_observations = amax_top_observations[year]
            amax_samples = amax_predictions[year]
            from scipy.stats import gaussian_kde

            density = gaussian_kde(amax_samples)

            x = np.linspace(0, max_global_precip + 2, 200)

            ax.plot(x, density(x), label="Sample amax distribution")
            ax.scatter(
                top_observations[0],
                0,
                s=100,
                color="r",
                marker="x",
                label="Max observation",
            )
            ax.scatter(
                top_observations[1:],
                np.zeros_like(top_observations[1:]),
                s=50,
                color="black",
                marker="x",
                label="Top observations",
            )
            ax.set_title(f"Annual max {self.output_variable} statistics, {year}")
            ax.set_ylabel("Prediction sample density")
        axs[0].legend()
        axs[-1].set_xlabel(f"Annual max {self.output_variable} ({self.units})")
        plt.savefig(self.output_dir / "annual_max_distributions.png")
        plt.close()
