"""
Preprocess gridded climate datasets and estimate local temperature acceleration.

This script:
1. loads HadCRUT5, NOAA GlobalTemp, Berkeley Earth, or ERA5 data;
2. converts the temperature fields to a common spatial representation;
3. adjusts the temperature anomaly baseline;
4. aggregates monthly observations to annual means;
5. estimates empirical prior parameters;
6. samples posterior distributions for a quadratic temperature model.

The generated files are used as inputs to grf.py.

Run this script after placing the required NetCDF files in the local data
directory.
"""

from pathlib import Path
import json

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.ndimage import zoom


# Repository paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Map projection used for spatial figures
projection = ccrs.Robinson()


def custom_latitude_formatter(lat, pos):
    """Format latitude labels using north and south notation."""
    direction = "N" if lat >= 0 else "S"
    return f"{abs(int(lat))}°{direction}"


def custom_longitude_formatter(lon, pos):
    """Format longitude labels using east and west notation."""
    direction = "E" if lon >= 0 else "W"
    return f"{abs(int(lon))}°{direction}"


def gibbs(
    n_samples,
    temperature,
    beta,
    alpha=None,
    mu=None,
    burnin=0.1,
):
    """
    Sample the posterior distribution of a quadratic temperature model.

    The fitted model is

        T_t = a_0 + a_1 t + a_2 t^2 + epsilon_t.

    Parameters
    ----------
    n_samples : int
        Total number of Gibbs-sampling iterations.
    temperature : array-like
        Annual temperature observations for one spatial grid cell.
    beta : array-like of length 4
        Scale parameters for the inverse-gamma prior distributions.
    alpha : array-like of length 4, optional
        Shape parameters for the inverse-gamma prior distributions.
    mu : array-like of length 3, optional
        Prior means for the quadratic regression coefficients.
    burnin : float, optional
        Fraction of initial samples discarded as burn-in.

    Returns
    -------
    numpy.ndarray
        Posterior samples arranged by parameter. The first three rows contain
        samples of a_0, a_1, and a_2, followed by the variance parameters.
    """
    if alpha is None:
        alpha = np.ones(4) * 3

    if mu is None:
        mu = np.array(
            [0.052689, 0.01346537, 0.00022735],
            dtype=float,
        )

    temperature = np.asarray(temperature, dtype=float)
    beta = np.asarray(beta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if not 0 <= burnin < 1:
        raise ValueError("burnin must be between 0 and 1.")

    initial_values = np.concatenate(
        (mu, beta / (alpha - 1))
    )

    samples = np.zeros(
        (n_samples, len(initial_values)),
        dtype=float,
    )
    samples[0] = initial_values

    time_index = np.arange(
        len(temperature),
        dtype=float,
    )

    design_matrix = np.column_stack(
        (
            np.ones_like(time_index),
            time_index,
            time_index**2,
        )
    )

    n_observations = len(temperature)

    for sample_index in range(1, n_samples):
        observation_variance = samples[
            sample_index - 1,
            3,
        ]

        coefficient_precision = np.diag(
            samples[sample_index - 1, 4:7]
        )

        posterior_covariance = np.linalg.inv(
            design_matrix.T
            @ design_matrix
            / observation_variance
            + coefficient_precision
        )

        posterior_mean = posterior_covariance @ (
            design_matrix.T
            @ temperature
            / observation_variance
            + coefficient_precision @ mu
        )

        samples[sample_index, 0:3] = (
            np.random.multivariate_normal(
                posterior_mean,
                posterior_covariance,
            )
        )

        fitted_temperature = (
            design_matrix
            @ samples[sample_index, 0:3]
        )

        residual_sum_squares = np.sum(
            (
                temperature
                - fitted_temperature
            )
            ** 2
        )

        samples[sample_index, 3] = 1 / np.random.gamma(
            alpha[0] + n_observations / 2,
            scale=1
            / (
                0.5 * residual_sum_squares
                + beta[0]
            ),
        )

        for coefficient_index in range(3):
            coefficient_difference = (
                samples[
                    sample_index,
                    coefficient_index,
                ]
                - mu[coefficient_index]
            )

            samples[
                sample_index,
                coefficient_index + 4,
            ] = 1 / np.random.gamma(
                alpha[coefficient_index + 1] + 0.5,
                scale=1
                / (
                    0.5 * coefficient_difference**2
                    + beta[coefficient_index + 1]
                ),
            )

    burnin_index = int(n_samples * burnin)

    return samples[burnin_index:].T


# %% Load selected climate dataset

# All datasets extend to the present except Berkeley Earth, which ends in 2024.
data_origin = "noaa"

# Supported options:
# "hadcrut", "noaa", "berkeley", and "era5".
data_path = DATA_DIR / f"data_grid_{data_origin}.nc"
result_dir = RESULTS_DIR / data_origin
result_dir.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    raise FileNotFoundError(
        f"Input dataset was not found: {data_path}"
    )

with nc.Dataset(str(data_path), mode="r") as dataset:
    if data_origin == "hadcrut":
        first_meas = 1850

        lat = np.asarray(
            dataset.variables["latitude"][:].data
        )
        lon = np.asarray(
            dataset.variables["longitude"][:].data
        )

        temperature_variable = dataset.variables[
            "tas_mean"
        ][:]

        data = np.asarray(
            temperature_variable.data,
            dtype=float,
        )

        masks = np.ma.getmaskarray(
            temperature_variable
        )

    elif data_origin == "noaa":
        first_meas = 1850

        lat = np.asarray(
            dataset.variables["lat"][:].data
        )

        # Shift NOAA longitudes to the range used by the other datasets.
        lon = np.asarray(
            dataset.variables["lon"][:].data
        ) - 180

        temperature_variable = dataset.variables[
            "anom"
        ][:]

        data = np.asarray(
            np.ma.filled(
                temperature_variable,
                np.nan,
            ),
            dtype=float,
        ).squeeze(axis=1)

    elif data_origin == "berkeley":
        first_meas = 1850

        lat = (
            np.asarray(
                dataset.variables["latitude"][:].data
            )
            .reshape(180 // 5, 5)
            .mean(axis=1)
        )

        lon = (
            np.asarray(
                dataset.variables["longitude"][:].data
            )
            .reshape(360 // 5, 5)
            .mean(axis=1)
        )

        temperature_variable = dataset.variables[
            "temperature"
        ][:]

        raw_data = np.asarray(
            np.ma.filled(
                temperature_variable,
                np.nan,
            ),
            dtype=float,
        )

        data = np.nanmean(
            raw_data.reshape(
                raw_data.shape[0],
                36,
                5,
                72,
                5,
            ),
            axis=(2, 4),
        )

    elif data_origin == "era5":
        first_meas = 1940

        raw_lat = np.asarray(
            dataset.variables["latitude"][:].data
        )
        raw_lon = np.asarray(
            dataset.variables["longitude"][:].data
        )

        lat = np.flipud(
            np.mean(
                [
                    raw_lat[
                        i * 21 - i:
                        (i + 1) * 21 - i
                    ]
                    for i in range(36)
                ],
                axis=1,
            )
        )

        lon = (
            np.mean(
                [
                    raw_lon[1:][
                        i * 19 + i:
                        (i + 1) * 19 + i
                    ]
                    for i in range(72)
                ],
                axis=1,
            )
            - 180
        )

        temperature_variable = dataset.variables[
            "t2m"
        ][:]

        raw_data = np.asarray(
            np.ma.filled(
                temperature_variable,
                np.nan,
            ),
            dtype=float,
        ) - 273.15

        data = np.zeros(
            (raw_data.shape[0], 36, 72),
            dtype=float,
        )

        for month_index, monthly_field in enumerate(
            raw_data
        ):
            monthly_field = np.flipud(
                monthly_field
            )

            for lat_index in range(36):
                lat_start = (
                    lat_index * 21 - lat_index
                )
                lat_stop = (
                    (lat_index + 1) * 21
                    - lat_index
                )

                for lon_index in range(72):
                    lon_start = (
                        lon_index * 19 + lon_index
                    )
                    lon_stop = (
                        (lon_index + 1) * 19
                        + lon_index
                    )

                    data[
                        month_index,
                        lat_index,
                        lon_index,
                    ] = np.nanmean(
                        monthly_field[
                            :,
                            :-1,
                        ][
                            lat_start:lat_stop,
                            lon_start:lon_stop,
                        ]
                    )

    else:
        raise ValueError(
            "data_origin must be 'hadcrut', 'noaa', "
            "'berkeley', or 'era5'."
        )


# %% Adjust the temperature reference baseline

# Mean over the 1961–1990 reference period.
current_sum = np.zeros_like(
    data[0],
    dtype=float,
)
current_count = np.zeros_like(
    data[0],
    dtype=float,
)

current_start = 12 * (
    1961 - first_meas
)
current_stop = 12 * (
    1991 - first_meas
)

for month_index in range(
    current_start,
    current_stop,
):
    if data_origin == "hadcrut":
        missing = masks[month_index]
    else:
        missing = np.isnan(
            data[month_index]
        )

    current_sum += np.where(
        missing,
        0.0,
        data[month_index],
    )
    current_count += ~missing

current_mean = np.divide(
    current_sum,
    current_count,
    out=np.full_like(
        current_sum,
        np.nan,
    ),
    where=current_count > 0,
)


# Mean over the first 50 years of datasets containing pre-industrial data.
if data_origin != "era5":
    preindustrial_sum = np.zeros_like(
        data[0],
        dtype=float,
    )
    preindustrial_count = np.zeros_like(
        data[0],
        dtype=float,
    )

    for month_index in range(12 * 50):
        if data_origin == "hadcrut":
            missing = masks[month_index]
        else:
            missing = np.isnan(
                data[month_index]
            )

        preindustrial_sum += np.where(
            missing,
            0.0,
            data[month_index],
        )
        preindustrial_count += ~missing

    preindustrial_mean = np.divide(
        preindustrial_sum,
        preindustrial_count,
        out=np.full_like(
            preindustrial_sum,
            np.nan,
        ),
        where=preindustrial_count > 0,
    )

else:
    preindustrial_mean = 0.0


# Preserve the baseline transformation used in the original analysis while
# calculating the two means with their correct, separate observation counts.
data = data - (
    preindustrial_mean
    + current_mean
)

# %% Plot annual mean temperature fields for selected years

years = [1980, 2000, 2010, 2025]

# Berkeley Earth does not contain a complete 2025 calendar year.
if data_origin == "berkeley":
    years.remove(2025)

annual_means = []

for year in years:
    month_start = (year - first_meas) * 12
    month_stop = (year + 1 - first_meas) * 12

    if month_stop > len(data):
        raise ValueError(
            f"The {data_origin} dataset does not contain a complete "
            f"calendar year for {year}."
        )

    annual_sum = np.zeros_like(
        data[0],
        dtype=float,
    )
    annual_count = np.zeros_like(
        data[0],
        dtype=float,
    )

    for month_index in range(month_start, month_stop):
        if data_origin == "hadcrut":
            missing = masks[month_index]
        else:
            missing = np.isnan(data[month_index])

        annual_sum += np.where(
            missing,
            0.0,
            data[month_index],
        )
        annual_count += ~missing

    annual_mean = np.divide(
        annual_sum,
        annual_count,
        out=np.full_like(
            annual_sum,
            np.nan,
        ),
        where=annual_count > 0,
    )

    annual_means.append(annual_mean)


# Use a common scale across all displayed years.
vmin = np.floor(
    np.nanmin(np.asarray(annual_means))
)
vmax = np.ceil(
    np.nanmax(np.asarray(annual_means))
)

n_columns = 2
n_rows = int(
    np.ceil(len(years) / n_columns)
)

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_columns,
    figsize=(10, 5 * n_rows / 2),
    subplot_kw={"projection": projection},
    squeeze=False,
)

# The latitude arrays are expected to be ordered from south to north.
image_origin = (
    "lower"
    if lat[0] < lat[-1]
    else "upper"
)

for plot_index, (year, annual_mean) in enumerate(
    zip(years, annual_means)
):
    row, column = divmod(
        plot_index,
        n_columns,
    )

    ax = axes[row, column]

    annual_mean_zoom = zoom(
        annual_mean,
        (2, 2),
        order=1,
    )

    image = ax.imshow(
        annual_mean_zoom,
        extent=[
            float(np.min(lon)),
            float(np.max(lon)),
            float(np.min(lat)),
            float(np.max(lat)),
        ],
        origin=image_origin,
        transform=ccrs.PlateCarree(),
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(
        f"Mean temperature anomaly, {year}"
    )

    ax.coastlines(
        color="black",
        linewidth=0.5,
    )

    ax.add_feature(
        cfeature.BORDERS,
        edgecolor="black",
        linewidth=0.5,
    )

    gridlines = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        linewidth=0.5,
        color="gray",
    )

    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xformatter = FuncFormatter(
        custom_longitude_formatter
    )
    gridlines.yformatter = FuncFormatter(
        custom_latitude_formatter
    )
    gridlines.ylocator = MaxNLocator(
        nbins=9
    )


# Hide unused axes when fewer than four years are displayed.
for plot_index in range(
    len(years),
    n_rows * n_columns,
):
    row, column = divmod(
        plot_index,
        n_columns,
    )
    axes[row, column].set_visible(False)


colorbar = fig.colorbar(
    image,
    ax=axes.ravel().tolist(),
    shrink=0.9,
    aspect=40,
    orientation="vertical",
    pad=0.05,
)

colorbar.set_label(
    "Temperature anomaly (°C)"
)

fig.suptitle(
    data_origin.upper()
)

# figure_path = result_dir / (
#     f"annual_temperature_fields_{data_origin}.pdf"
# )
# fig.savefig(
#     figure_path,
#     bbox_inches="tight",
# )

plt.show()

# %% Helper functions for empirical prior estimation

# %% Helper functions for empirical prior estimation

def bootstrapping(data_values, n_samples):
    """
    Generate bootstrap temperature series by resampling model residuals.

    Each bootstrap series is constructed by adding resampled residuals to the
    fitted quadratic temperature trajectory.

    Parameters
    ----------
    data_values : sequence
        Sequence containing the time index, observed temperature series, and
        fitted temperature series.
    n_samples : int
        Number of bootstrap series to generate.

    Returns
    -------
    numpy.ndarray
        Bootstrap temperature series with shape
        (n_samples, n_observations).
    """
    _, observed, fitted = data_values

    observed = np.asarray(observed, dtype=float)
    fitted = np.asarray(fitted, dtype=float)

    if observed.shape != fitted.shape:
        raise ValueError(
            "Observed and fitted series must have the same shape."
        )

    residuals = observed - fitted

    bootstrap_data = np.zeros(
        (n_samples, len(observed)),
        dtype=float,
    )

    for sample_index in range(n_samples):
        sampled_residuals = np.random.choice(
            residuals,
            size=len(residuals),
            replace=True,
        )

        bootstrap_data[sample_index] = (
            fitted + sampled_residuals
        )

    return bootstrap_data


def poly(x, coefficients):
    """
    Evaluate a polynomial with coefficients ordered by increasing power.

    Parameters
    ----------
    x : array-like
        Points at which the polynomial is evaluated.
    coefficients : array-like
        Polynomial coefficients ordered as the constant, linear, quadratic,
        and any subsequent higher-order terms.

    Returns
    -------
    numpy.ndarray
        Evaluated polynomial values.
    """
    x = np.asarray(x, dtype=float)
    coefficients = np.asarray(
        coefficients,
        dtype=float,
    )

    return np.vander(
        x,
        N=len(coefficients),
        increasing=True,
    ) @ coefficients

# %% Extract annual data and estimate empirical prior parameters

# %% Extract annual data and estimate empirical prior parameters

start_year = 1970

# End years are exclusive. For example, 2026 includes annual observations
# from 1970 through 2025.
end_years = [
    1990,
    1995,
    2000,
    2005,
    2010,
    2015,
    2020,
    2026,
]

n_bootstrap_samples = 1000

result_dir = RESULTS_DIR / data_origin
result_dir.mkdir(parents=True, exist_ok=True)

coordinates = [
    f"{latitude} {longitude}"
    for latitude in lat
    for longitude in lon
]

monthly_start = (
    start_year - first_meas
) * 12


for end_year in end_years:
    n_years = end_year - start_year
    expected_months = n_years * 12

    monthly_stop = (
        end_year - first_meas
    ) * 12

    if monthly_start < 0:
        raise ValueError(
            f"The selected start year {start_year} precedes the first "
            f"measurement year {first_meas} for {data_origin}."
        )

    if monthly_stop > len(data):
        raise ValueError(
            f"The {data_origin} dataset does not contain complete data "
            f"through December {end_year - 1}."
        )

    period_data = np.asarray(
        data[monthly_start:monthly_stop],
        dtype=float,
    )

    if data_origin == "hadcrut":
        period_masks = np.asarray(
            masks[monthly_start:monthly_stop],
            dtype=bool,
        )
    else:
        period_masks = None

    data_dict = {}

    for lat_index, latitude in enumerate(lat):
        for lon_index, longitude in enumerate(lon):
            coordinate = f"{latitude} {longitude}"

            monthly_values = period_data[
                :,
                lat_index,
                lon_index,
            ].copy()

            if period_masks is not None:
                monthly_values = np.where(
                    period_masks[
                        :,
                        lat_index,
                        lon_index,
                    ],
                    np.nan,
                    monthly_values,
                )

            if len(monthly_values) != expected_months:
                raise ValueError(
                    f"Expected {expected_months} monthly observations for "
                    f"{coordinate}, but found {len(monthly_values)}."
                )

            annual_months = monthly_values.reshape(
                n_years,
                12,
            )

            # Interpolate within a calendar year when at most three monthly
            # observations are missing. Years with four or more missing
            # observations remain incomplete.
            for year_index in range(n_years):
                missing = np.isnan(
                    annual_months[year_index]
                )
                n_missing = np.count_nonzero(
                    missing
                )

                if 0 < n_missing <= 3:
                    month_index = np.arange(12)
                    observed = ~missing

                    annual_months[year_index] = np.interp(
                        month_index,
                        month_index[observed],
                        annual_months[
                            year_index,
                            observed,
                        ],
                    )

            # np.mean intentionally returns NaN for years that remain
            # incomplete after interpolation.
            annual_values = np.mean(
                annual_months,
                axis=1,
            )

            data_dict[coordinate] = (
                annual_values.astype(float).tolist()
            )

    data_output_path = result_dir / (
        f"data_{data_origin}_{start_year}_{end_year}.txt"
    )

    with data_output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(data_dict, file)

    print(
        f"Annual data saved to: {data_output_path}"
    )

    # Estimate grid-cell-specific prior scale parameters.
    beta = {
        coordinate: None
        for coordinate in coordinates
    }

    coefficient_sum = np.zeros(
        3,
        dtype=float,
    )
    valid_grid_count = 0

    for coordinate in coordinates:
        temperature = np.asarray(
            data_dict[coordinate],
            dtype=float,
        )

        # Prior quantities are estimated only for complete annual series.
        if not np.all(
            np.isfinite(temperature)
        ):
            continue

        time_index = np.arange(
            len(temperature),
            dtype=float,
        )

        # np.polyfit returns coefficients in decreasing order. Reverse them
        # to obtain [intercept, linear coefficient, quadratic coefficient].
        coefficients = np.flip(
            np.polyfit(
                time_index,
                temperature,
                deg=2,
            )
        )

        fitted_temperature = poly(
            time_index,
            coefficients,
        )

        coefficient_sum += coefficients
        valid_grid_count += 1

        variance_estimates = np.zeros(
            4,
            dtype=float,
        )

        # Residual variance of the fitted quadratic model.
        variance_estimates[3] = np.mean(
            (
                temperature
                - fitted_temperature
            )
            ** 2
        )

        bootstrap_data = bootstrapping(
            [
                time_index,
                temperature,
                fitted_temperature,
            ],
            n_samples=n_bootstrap_samples,
        )

        bootstrap_coefficients = np.zeros(
            (
                n_bootstrap_samples,
                3,
            ),
            dtype=float,
        )

        for sample_index, bootstrap_series in enumerate(
            bootstrap_data
        ):
            bootstrap_coefficients[
                sample_index
            ] = np.flip(
                np.polyfit(
                    time_index,
                    bootstrap_series,
                    deg=2,
                )
            )

        # Population variance preserves the calculation used in the
        # original implementation.
        variance_estimates[:3] = np.var(
            bootstrap_coefficients,
            axis=0,
            ddof=0,
        )

        beta[coordinate] = (
            2.0 * variance_estimates
        ).tolist()

    if valid_grid_count == 0:
        raise ValueError(
            f"No complete grid-cell series were available for "
            f"{start_year}-{end_year}."
        )

    prior_mean = (
        coefficient_sum
        / valid_grid_count
    )

    beta["mu"] = prior_mean.tolist()

    beta_output_path = result_dir / (
        f"beta_{data_origin}_{start_year}_{end_year}.txt"
    )

    with beta_output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(beta, file)

    print(
        f"Prior parameters saved to: {beta_output_path}"
    )
    print(
        f"Complete grid cells: "
        f"{valid_grid_count}/{len(coordinates)}"
    )

# %% Run Gibbs sampling

n_gibbs_samples = 100_000

for end_year in end_years:
    data_input_path = result_dir / (
        f"data_{data_origin}_{start_year}_{end_year}.txt"
    )

    beta_input_path = result_dir / (
        f"beta_{data_origin}_{start_year}_{end_year}.txt"
    )

    with data_input_path.open("r", encoding="utf-8") as file:
        data_dict = json.load(file)

    with beta_input_path.open("r", encoding="utf-8") as file:
        beta = json.load(file)

    post_means = dict.fromkeys(coordinates, None)
    prob_g_0 = dict.fromkeys(coordinates, None)

    prior_mean = np.asarray(
        beta["mu"],
        dtype=float,
    )

    for coordinate in coordinates:
        temperature = np.asarray(
            data_dict[coordinate],
            dtype=float,
        )

        if not np.all(np.isfinite(temperature)):
            continue

        beta_coordinate = beta.get(coordinate)

        if beta_coordinate is None:
            continue

        simulation = gibbs(
            n_gibbs_samples,
            temperature,
            np.asarray(beta_coordinate, dtype=float),
            mu=prior_mean,
        )

        post_means[coordinate] = np.mean(
            simulation,
            axis=1,
        ).tolist()

        prob_g_0[coordinate] = np.mean(
            simulation >= 0,
            axis=1,
        ).tolist()

    post_means_path = result_dir / (
        f"post_means_{data_origin}_{start_year}_{end_year}.txt"
    )

    probability_path = result_dir / (
        f"prob_g_0_{data_origin}_{start_year}_{end_year}.txt"
    )

    with post_means_path.open("w", encoding="utf-8") as file:
        json.dump(post_means, file)

    with probability_path.open("w", encoding="utf-8") as file:
        json.dump(prob_g_0, file)

    print(
        f"Posterior results saved for "
        f"{start_year}-{end_year}"
    )


# %% Create trace plots for selected grid cells

diagnostic_origin = "era5"
diagnostic_start_year = 1970
diagnostic_end_year = 1990

diagnostic_result_dir = RESULTS_DIR / diagnostic_origin
diagnostic_figure_dir = diagnostic_result_dir / "traceplots"
diagnostic_figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)

diagnostic_data_path = diagnostic_result_dir / (
    f"data_{diagnostic_origin}_"
    f"{diagnostic_start_year}_{diagnostic_end_year}.txt"
)

diagnostic_beta_path = diagnostic_result_dir / (
    f"beta_{diagnostic_origin}_"
    f"{diagnostic_start_year}_{diagnostic_end_year}.txt"
)

with diagnostic_data_path.open("r", encoding="utf-8") as file:
    diagnostic_data = json.load(file)

with diagnostic_beta_path.open("r", encoding="utf-8") as file:
    diagnostic_beta = json.load(file)

selected_latitudes = [
    "57.5",
    "-2.5",
    "-17.5",
    "-62.5",
]

selected_longitudes = [
    "87.5",
    "32.5",
    "-152.5",
    "62.5",
]

trace_steps = 100

for location_index, (latitude, longitude) in enumerate(
    zip(
        selected_latitudes,
        selected_longitudes,
    )
):
    coordinate = f"{latitude} {longitude}"

    if coordinate not in diagnostic_data:
        print(
            f"Skipping trace plot because coordinate "
            f"{coordinate} was not found."
        )
        continue

    temperature = np.asarray(
        diagnostic_data[coordinate],
        dtype=float,
    )

    beta_coordinate = diagnostic_beta.get(coordinate)

    if (
        not np.all(np.isfinite(temperature))
        or beta_coordinate is None
    ):
        continue

    simulation = gibbs(
        1000,
        temperature,
        beta=np.asarray(
            beta_coordinate,
            dtype=float,
        ),
        mu=np.asarray(
            diagnostic_beta["mu"],
            dtype=float,
        ),
        burnin=0,
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(6, 8),
        sharex=True,
    )

    for parameter_index in range(3):
        axes[parameter_index].plot(
            simulation[parameter_index, :trace_steps],
            "o",
            label=rf"Trace of $a_{parameter_index}$",
        )

        axes[parameter_index].plot(
            simulation[parameter_index, :trace_steps],
            "b--",
        )

        axes[parameter_index].legend(
            loc="upper right"
        )

    axes[0].set_title(
        f"Trace plot at lat = {latitude}, "
        f"lon = {longitude}"
    )

    coefficient_trace_path = diagnostic_figure_dir / (
        f"{location_index}_coefficients.pdf"
    )

    fig.tight_layout()
    fig.savefig(
        coefficient_trace_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(6, 8),
        sharex=True,
    )

    axes[0].plot(
        simulation[3, :trace_steps],
        "o",
        label=r"Trace of $\sigma^2$",
    )

    axes[0].plot(
        simulation[3, :trace_steps],
        "b--",
    )

    axes[0].legend(
        loc="upper right"
    )

    for variance_index in range(4, 7):
        axis_index = variance_index - 3
        parameter_index = variance_index - 4

        axes[axis_index].plot(
            simulation[variance_index, :trace_steps],
            "o",
            label=(
                rf"Trace of "
                rf"$\sigma^2_{{{parameter_index}}}$"
            ),
        )

        axes[axis_index].plot(
            simulation[variance_index, :trace_steps],
            "b--",
        )

        axes[axis_index].legend(
            loc="upper right"
        )

    axes[0].set_title(
        f"Trace plot at lat = {latitude}, "
        f"lon = {longitude}"
    )

    variance_trace_path = diagnostic_figure_dir / (
        f"{location_index}_variances.pdf"
    )

    fig.tight_layout()
    fig.savefig(
        variance_trace_path,
        bbox_inches="tight",
    )
    plt.close(fig)


# %% Plot fitted polynomial trajectories at selected grid cells

fit_origin = "era5"
fit_start_year = 1970
fit_end_year = 2026

fit_result_dir = RESULTS_DIR / fit_origin
fit_figure_dir = fit_result_dir / "fitted_trajectories"
fit_figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)

fit_data_path = fit_result_dir / (
    f"data_{fit_origin}_{fit_start_year}_{fit_end_year}.txt"
)

fit_post_means_path = fit_result_dir / (
    f"post_means_{fit_origin}_"
    f"{fit_start_year}_{fit_end_year}.txt"
)

with fit_data_path.open("r", encoding="utf-8") as file:
    fit_data = json.load(file)

with fit_post_means_path.open("r", encoding="utf-8") as file:
    fit_post_means = json.load(file)

acceleration_values = []

for latitude, longitude in zip(
    selected_latitudes,
    selected_longitudes,
):
    coordinate = f"{latitude} {longitude}"

    if (
        coordinate not in fit_data
        or coordinate not in fit_post_means
        or fit_post_means[coordinate] is None
    ):
        print(
            f"Skipping fitted trajectory because coordinate "
            f"{coordinate} is unavailable."
        )
        continue

    temperature = np.asarray(
        fit_data[coordinate],
        dtype=float,
    )

    coefficients = np.asarray(
        fit_post_means[coordinate][:3],
        dtype=float,
    )

    if not np.all(np.isfinite(temperature)):
        continue

    acceleration_values.append(
        coefficients[2]
    )

    time_index = np.arange(
        len(temperature),
        dtype=float,
    )

    fitted_temperature = (
        coefficients[0]
        + coefficients[1] * time_index
        + coefficients[2] * time_index**2
    )

    calendar_years = (
        fit_start_year
        + time_index.astype(int)
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        calendar_years,
        temperature,
        "bo",
        label="Measurements",
    )

    ax.plot(
        calendar_years,
        temperature,
        "b--",
    )

    ax.plot(
        calendar_years,
        fitted_temperature,
        "r",
        label="Fitted polynomial",
    )

    ax.set_title(
        f"lat = {latitude}, lon = {longitude}"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature anomaly")
    ax.legend()

    fig.tight_layout()

    fitted_figure_path = fit_figure_dir / (
        f"data_and_poly_{latitude}_{longitude}.pdf"
    )

    fig.savefig(
        fitted_figure_path,
        bbox_inches="tight",
    )

    plt.close(fig)