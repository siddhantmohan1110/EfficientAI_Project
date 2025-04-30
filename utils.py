import numpy as np
from scipy.stats import kstest, lognorm, norm


def qq_plot(grad):

    grad = grad.cpu().numpy()
    grad = grad[np.isfinite(grad)]

    # 1. Remove outliers
    mean_grad = np.mean(grad)
    std_grad = np.std(grad)

    # Keep only gradients within ±3 standard deviations
    mask = (grad > (mean_grad - 3 * std_grad)) & (grad < (mean_grad + 3 * std_grad))
    grad_clean = grad[mask]

    percent_retained = (grad_clean.shape[0] / grad.shape[0]) * 100

    # Step 1: Fit a log-normal distribution to your data
    shape, loc, scale = lognorm.fit(grad_clean, floc=0)  # fixing loc=0 is common

    # 2. Generate theoretical quantiles
    # 'ppf' = percent point function = inverse CDF
    probs = np.linspace(0.01, 0.99, 100)
    theoretical_quantiles = lognorm.ppf(probs, shape, loc=loc, scale=scale)

    # 3. Get empirical quantiles (sort the data)
    empirical_quantiles = np.quantile(grad, probs)

    return theoretical_quantiles, empirical_quantiles, percent_retained


def qq_plot_outliers(grad):

    grad = grad.cpu().numpy()
    grad = grad[np.isfinite(grad)]

    # 1. Fit the full data to a log-normal
    shape, loc, scale = lognorm.fit(grad, floc=0)

    # 2. Set up probabilities
    probs = np.linspace(0.01, 0.99, 500)  # finer granularity

    # 3. Get theoretical and empirical quantiles
    theoretical_quantiles = lognorm.ppf(probs, shape, loc=loc, scale=scale)
    empirical_quantiles = np.quantile(grad, probs)

    # 4. Compute deviations
    deviations = np.abs(empirical_quantiles - theoretical_quantiles)

    # 5. Set a threshold for acceptable deviation
    # For example, allow 5% deviation relative to theoretical quantile
    relative_tolerance = 0.05  # 5%

    # Compute acceptable deviation based on theoretical
    acceptable_deviation = relative_tolerance * theoretical_quantiles

    # Find which quantiles are "acceptable"
    acceptable = deviations <= acceptable_deviation

    # 6. Find bounds of "good" quantiles
    lower_bound = empirical_quantiles[acceptable][0]
    upper_bound = empirical_quantiles[acceptable][-1]

    # 7. Mask original gradients within bounds
    grad_clean = grad[(grad >= lower_bound) & (grad <= upper_bound)]

    # 8. Percentage retained
    percent_retained = (grad_clean.shape[0] / grad.shape[0]) * 100

    # 9. Refit cleaned gradients
    shape_clean, loc_clean, scale_clean = lognorm.fit(grad_clean, floc=0)
    theoretical_quantiles_clean = lognorm.ppf(probs, shape_clean, loc=loc_clean, scale=scale_clean)
    empirical_quantiles_clean = np.quantile(grad, probs)

    return theoretical_quantiles_clean, empirical_quantiles_clean, percent_retained



def KS_test_normal(grad, alpha = 0.05) -> bool:

    grad = grad.cpu().numpy()
    grad = grad[np.isfinite(grad)]

    # Step 1: Fit a log-normal distribution to your data
    mu, std = norm.fit(grad, floc=0)  # fixing loc=0 is common

    # Step 2: Perform the KS test using the fitted parameters
    ks_stat, p_value = kstest(grad, 'norm', args=(mu, std))


    # Step 3: Print results
    # print("Kolmogorov–Smirnov Test for Log-normality")
    # print(f"KS Statistic: {ks_stat:.4f}")
    # print(f"P-value: {p_value:.4f}")


    if p_value < alpha:
        print("❌ Reject the null hypothesis: grad does NOT follow a log-normal distribution with p_value=", p_value)
        return False
    else:
        print("✅ Fail to reject the null hypothesis: grad MAY follow a log-normal distribution with p_value=", p_value)
        return True

    return p_value


