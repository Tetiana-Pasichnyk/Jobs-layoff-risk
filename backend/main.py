import os
from load import load_data, clean_data, encode_target
from stats_analysis import (
    run_eda,
    detect_outliers,
    correlation_analysis,
    chi_square_analysis,
    statistical_tests
)
from visualization import create_plots
from reporting import build_html, export_pdf


# ==============================================================================
# KONFIGURATION
# ==============================================================================

dateiname = "../data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv"

OUTPUT_DIR = "output_analysis"

numerische_spalten = [
    'Age', 'Years_of_Experience', 'Routine_Task_Percentage',
    'Creativity_Requirement', 'Human_Interaction_Level',
    'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
    'Tasks_Automated_Percentage', 'AI_Training_Hours'
]

kategorische_spalten = [
    'Education_Level', 'Industry', 'Job_Role',
    'Company_Size', 'Job_Level', 'AI_Adoption_Level', 'Layoff_Risk'
]


# ==============================================================================
# PIPELINE
# ==============================================================================

print("\n================ PIPELINE STARTET ================\n")

# 1. LADEN + BEREINIGEN
df = load_data(dateiname)
df = clean_data(df, numerische_spalten, kategorische_spalten)
df = encode_target(df)

print("\n[SCHRITT] Daten erfolgreich geladen und bereinigt\n")


# 2. EXPLORATIVE DATENANALYSE (EDA)
desc = run_eda(df, numerische_spalten)
outliers = detect_outliers(df, numerische_spalten, desc)

print("\n[SCHRITT] EDA abgeschlossen")
print(desc[["mean", "std"]].head())


# 3. STATISTISCHE ANALYSE
corr = correlation_analysis(df, numerische_spalten)
chi_df = chi_square_analysis(df, kategorische_spalten)
stats = statistical_tests(df, numerische_spalten)

print("\n[SCHRITT] Statistische Tests abgeschlossen")
print("Bevorzugte Methode:", stats["method"])
print("Konfidenzintervall (Routine):", stats["ci_routine"])
print("Konfidenzintervall (Automated):", stats["ci_automated"])

# 4. VISUALISIERUNG
plots = create_plots(df, chi_df, numerische_spalten, output_dir=OUTPUT_DIR)
print("\n[SCHRITT] Visualisierungen erstellt")


# 5. BERICHT (HTML + PDF)
html = build_html(
    desc_stats=desc,
    outliers_dict=outliers,
    ci_routine=stats["ci_routine"],
    ci_automated=stats["ci_automated"],
    shapiro_html="\n".join(
        [f"<li>{col}: {p:.4f}</li>" for col, p in stats["shapiro"]]
    ),
    pearson_r=stats["pearson"][0],
    p_pearson=stats["pearson"][1],
    spearman_r=stats["spearman"][0],
    p_spearman=stats["spearman"][1],
    spearman_ai=stats["spearman_ai"][0],
    p_spearman_ai=stats["spearman_ai"][1],
    z_stat=stats["ztest"][0],
    p_z=stats["ztest"][1],
    img_scatter=plots["scatter"],
    img_heat=plots["heatmap"],
    img_chi=plots["chi"],
    empfohlenes_verfahren=stats["method"]
)

pdf_pfad = os.path.join(OUTPUT_DIR, "Analysebericht_KI_Arbeitsmarkt.pdf")
export_pdf(html, pdf_pfad)


# ==============================================================================
# ENDGÜLTIGE AUSGABE
# ==============================================================================

print("\n================ PIPELINE BEENDET ================\n")

print("Datensatzgröße:", df.shape)
print("Ausreißer-Zusammenfassung:", {k: v for k, v in list(outliers.items())[:3]})
print("Korrelationsauszug:")
print(corr["Layoff_Risk_Numeric"].sort_values(ascending=False).head())

print("\nBericht gespeichert: Analysebericht_KI_Arbeitsmarkt.pdf")



