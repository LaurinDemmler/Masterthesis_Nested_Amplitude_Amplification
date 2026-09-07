from scipy import stats

n = 5  # trials
k = 5  # successes
alpha = 0.01  # for 99% CI

lower = stats.beta.ppf(alpha/2, k, n-k+1)
upper = stats.beta.ppf(1-alpha/2, k+1, n-k) if k < n else 1.0

print(f"{(1-alpha)*100:.0f}% CI: [{lower:.3f}, {upper:.3f}]")