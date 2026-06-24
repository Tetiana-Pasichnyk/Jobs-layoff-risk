import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ztest  
from scipy.stats import norm                    
from scipy.stats import chi2_contingency, shapiro, pearsonr, spearmanr


def run_eda(df, num_cols):
    desc = df[num_cols].describe().T
    desc["variance"] = df[num_cols].var()

    print("[EDA] Deskriptive Statistiken berechnet")

    return desc


def detect_outliers(df, num_cols, desc_stats):
    outliers = {}

    for col in num_cols:
        q1 = desc_stats.loc[col, "25%"]
        q3 = desc_stats.loc[col, "75%"]
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers[col] = df[(df[col] < lower) | (df[col] > upper)].shape[0]

    print("[OUTLIERS] Ausreißer-Erkennung abgeschlossen")

    return outliers


def correlation_analysis(df, num_cols):
    corr = df[num_cols + ["Layoff_Risk_Numeric"]].corr(method="spearman")

    print("[KORRELATION] Spearman-Korrelation berechnet")

    return corr


def chi_square_analysis(df, cat_cols):
    results = []

    for col in cat_cols:
        if col == "Layoff_Risk":
            continue

        table = pd.crosstab(df[col], df["Layoff_Risk"])
        chi2, p, _, _ = chi2_contingency(table)

        n = table.sum().sum()
        min_dim = min(table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        results.append((col, chi2, cramers_v, p))

    chi_df = pd.DataFrame(results, columns=["Feature", "Chi2", "Cramers_V", "p_value"])
    chi_df = chi_df.sort_values("Chi2", ascending=False)

    print("[CHI²] Kategoriale Zusammenhänge berechnet")

    return chi_df


def statistical_tests(df, num_cols):

    sample_size = min(len(df), 5000)
    shapiro_results = []

    for col in num_cols:
        sample = df[col].sample(sample_size, random_state=42)
        _, p = shapiro(sample)
        shapiro_results.append((col, p))

    print("[SHAPIRO] Normalitätstests abgeschlossen")

    all_normal = all(p > 0.05 for _, p in shapiro_results)
    method = "Pearson" if all_normal else "Spearman"

    # Korrelationen
    pearson_r, p_pearson = pearsonr(
        df["Routine_Task_Percentage"],
        df["Tasks_Automated_Percentage"]
    )

    spearman_r, p_spearman = spearmanr(
        df["Routine_Task_Percentage"],
        df["Tasks_Automated_Percentage"]
    )

    spearman_ai, p_spearman_ai = spearmanr(
        df["AI_Usage_Hours_Per_Week"],
        df["Tasks_Automated_Percentage"]
    )

    # Z-Test
    high = df[df["Layoff_Risk"] == "High"]["Tasks_Automated_Percentage"]
    low = df[df["Layoff_Risk"] == "Low"]["Tasks_Automated_Percentage"]
    z_stat, p_z = ztest(high, low)

    # Konfidenzintervall
    n = len(df)
  
    mean_routine = df["Routine_Task_Percentage"].mean()
    std_routine = df["Routine_Task_Percentage"].std(ddof=1)
    ci_routine = norm.interval(0.95, loc=mean_routine, scale=std_routine / np.sqrt(n))

    mean_automated = df["Tasks_Automated_Percentage"].mean()
    std_automated = df["Tasks_Automated_Percentage"].std(ddof=1)
    ci_automated = norm.interval(0.95, loc=mean_automated, scale=std_automated / np.sqrt(n))
   

    print("[STATS] Hypothesentests abgeschlossen")
    return {
        "shapiro": shapiro_results,
        "method": method,
        "pearson": (pearson_r, p_pearson),
        "spearman": (spearman_r, p_spearman),
        "spearman_ai": (spearman_ai, p_spearman_ai),
        "ztest": (z_stat, p_z),
        "ci_routine": ci_routine,      
        "ci_automated": ci_automated
    }