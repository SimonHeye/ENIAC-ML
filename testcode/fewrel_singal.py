import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway

# 1. Prepare data
x = np.array([5, 10, 15])
ZETT = np.array([30.71, 27.79, 26.17])
PCRED = np.array([22.67, 24.91, 25.14])
ZS_SKA = np.array([32.86, 34.03, 23.86])
AML_EIKC = np.array([44.28, 42.14, 43.10])

# Calculate standard deviation for shading (example values)
ZETT_std = np.std(ZETT) / 2
PCRED_std = np.std(PCRED) / 2
ZS_SKA_std = np.std(ZS_SKA) / 2
AML_EIKC_std = np.std(AML_EIKC) / 2

# Perform ANOVA
anova_result = f_oneway(ZETT, PCRED, ZS_SKA, AML_EIKC)
print("ANOVA F-statistic:", anova_result.statistic)
print("ANOVA p-value:", anova_result.pvalue)

# 2. Create plot
plt.figure(dpi=100)

# 3. Plot lines with shaded areas
plt.plot(x, ZETT, label='ZETT', color='#EBA782', linestyle='-', linewidth=2)
# plt.fill_between(x, ZETT - ZETT_std, ZETT + ZETT_std, color='#EBA782', alpha=0.3)

plt.plot(x, PCRED, label='PCRED', color='#549F9A', linestyle='-', linewidth=2)
# plt.fill_between(x, PCRED - PCRED_std, PCRED + PCRED_std, color='#549F9A', alpha=0.3)

plt.plot(x, ZS_SKA, label='ZS_SKA', color='#4370B4', linestyle='-', linewidth=2)
# plt.fill_between(x, ZS_SKA - ZS_SKA_std, ZS_SKA + ZS_SKA_std, color='#4370B4', alpha=0.3)

plt.plot(x, AML_EIKC, label='AML_EIKC', color='#C30078', linestyle='-', linewidth=2)
# plt.fill_between(x, AML_EIKC - AML_EIKC_std, AML_EIKC + AML_EIKC_std, color='#C30078', alpha=0.3)

# Plot scatter points
plt.scatter(x, ZETT, color='#EBA782', s=20)
plt.scatter(x, PCRED, color='#549F9A', s=20)
plt.scatter(x, ZS_SKA, color='#4370B4', s=20)
plt.scatter(x, AML_EIKC, color='#C30078', s=20)

# 4. Add annotations
plt.legend(fontsize=12)
_xtick_labels = ['5', '10', '15']
plt.xticks(x, _xtick_labels, fontsize=14)
plt.xlabel('Unseen Relations', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

# 5. Show plot
plt.show()