import numpy as np
from scipy.stats import kstest, lognorm


def KS_test(grad, alpha = 0.05) -> bool:

    # Step 1: Fit a log-normal distribution to your data
    shape, loc, scale = lognorm.fit(grad, floc=0)  # fixing loc=0 is common

    # Step 2: Perform the KS test using the fitted parameters
    ks_stat, p_value = kstest(grad, 'lognorm', args=(shape, loc, scale))

    # Step 3: Print results
    print("Kolmogorov–Smirnov Test for Log-normality")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4f}")


    if p_value < alpha:
        print("❌ Reject the null hypothesis: grad does NOT follow a log-normal distribution.")
        return False
    else:
        print("✅ Fail to reject the null hypothesis: grad MAY follow a log-normal distribution.")
        return True

