from src.paths import OUTPUT_DIR
from src.statistics.load import clean_data, encode_target, load_data
from src.statistics.stats_analysis import (
    chi_square_analysis,
    correlation_analysis,
    detect_outliers,
    run_eda,
    statistical_tests,
)
from src.statistics.visualization import create_plots

NUMERIC_COLUMNS = [
    'Age', 'Years_of_Experience', 'Routine_Task_Percentage',
    'Creativity_Requirement', 'Human_Interaction_Level',
    'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
    'Tasks_Automated_Percentage', 'AI_Training_Hours'
]

CATEGORICAL_COLUMNS = [
    'Education_Level', 'Industry', 'Job_Role',
    'Company_Size', 'Job_Level', 'AI_Adoption_Level', 'Layoff_Risk'
]


def main():
    print("\n================ PIPELINE STARTET ================\n")

    df = load_data()
    df = clean_data(df, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
    df = encode_target(df)
    print("\n[SCHRITT] Daten erfolgreich geladen und bereinigt\n")

    desc = run_eda(df, NUMERIC_COLUMNS)
    outliers = detect_outliers(df, NUMERIC_COLUMNS, desc)
    print("\n[SCHRITT] EDA abgeschlossen")
    print(desc[["mean", "std"]].head())

    corr = correlation_analysis(df, NUMERIC_COLUMNS)
    chi_df = chi_square_analysis(df, CATEGORICAL_COLUMNS)
    stats = statistical_tests(df, NUMERIC_COLUMNS)
    print("\n[SCHRITT] Statistische Tests abgeschlossen")
    print("Bevorzugte Methode:", stats["method"])
    print("Konfidenzintervall (Routine):", stats["ci_routine"])
    print("Konfidenzintervall (Automated):", stats["ci_automated"])

    create_plots(df, chi_df, NUMERIC_COLUMNS, output_dir=str(OUTPUT_DIR))
    print("\n[SCHRITT] Visualisierungen erstellt")

    print("\n================ PIPELINE BEENDET ================\n")
    print("Datensatzgröße:", df.shape)
    print("Ausreißer-Zusammenfassung:", {k: v for k, v in list(outliers.items())[:3]})
    print("Korrelationsauszug:")
    print(corr["Layoff_Risk_Numeric"].sort_values(ascending=False).head())


if __name__ == "__main__":
    main()
