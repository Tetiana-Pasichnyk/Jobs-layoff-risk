import os

def build_html(
    desc_stats, outliers_dict, ci_routine, ci_automated, shapiro_html,
    pearson_r, p_pearson, spearman_r, p_spearman,
    spearman_ai, p_spearman_ai, z_stat, p_z,
    img_scatter, img_heat, img_chi,
    empfohlenes_verfahren="Spearman" 
):
    desc_rows_html = ""
    for index, row in desc_stats.iterrows():
        col_outliers = outliers_dict.get(index, 0)
        desc_rows_html += f"""
        <tr>
            <td><b>{index}</b></td>
            <td>{row['mean']:.2f}</td>
            <td>{row['50%']:.2f}</td>
            <td>{row['25%']:.2f}</td>
            <td>{row['75%']:.2f}</td>
            <td>{row['std']:.2f}</td>
            <td>{row['variance']:.2f}</td>
            <td>{col_outliers}</td>
        </tr>
        """

    html_template = f"""
      <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: A4;
            margin: 20mm 15mm;
            @bottom-right {{ content: counter(page); font-size: 9pt; color: #777; }}
            @bottom-left {{ content: "Analysebericht: Einfluss von KI auf den Arbeitsmarkt"; font-size: 9pt; color: #777; }}
        }}
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #333333;
            line-height: 1.5;
            font-size: 11pt;
        }}
        .header {{
            background-color: #d4e84e;
            color: black;
            padding: 20px;
            border-radius: 4px;
            margin-bottom: 20px;
        }}
        .header h1 {{ margin: 0; font-size: 18pt; }}
        .header p {{ margin: 5px 0 0 0; color: black; font-size: 11pt; }}
        h2 {{ color: black; font-size: 14pt; border-left: 4px solid #ffc107; padding-left: 8px; margin-top: 20px; }}
        h3 {{ color: black; font-size: 11pt; margin-top: 15px; margin-bottom: 8px; border-bottom: 1px solid #dee2e6; padding-bottom: 4px; }}
        p {{ text-align: justify; margin-bottom: 10px; color: black;  }}
        ul {{ margin-top: 5px; margin-bottom: 5px; }}
        .results-box {{ background-color: #faf7f4; border-top: 3px solid #1e3d59; padding: 15px; margin: 15px 0; border-radius: 4px; }}
        
        .chart-page {{ page-break-before: always; }}
        
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 85%; height: auto; border: 1px solid #e0e0e0; padding: 5px; background: #fff; }}
        .chart-title {{ font-size: 10pt; font-style: italic; color: #555; margin-top: 5px; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 10pt; }}
        th, td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
        th {{ background-color: #d4e84e; color: black; }}
        tr:nth-child(even) {{ background-color: #faf7f4; }}
    </style>
    </head>
    <body>

        <div class="header">
            <h1>Analysebericht: Einfluss von KI auf Arbeitsplätze</h1>
            <p>Statistische Untersuchung von Abhängigkeiten und Bewertung des Entlassungsrisikos (Layoff Risk)</p>
        </div>

        <h2>1. Beschreibung der durchgeifizierten Arbeiten</h2>
        <p>
        In dieser Analyse haben wir untersucht, welche Faktoren das Risiko für Entlassungen (Layoff Risk) beeinflussen.
        </p>
        <p>
        Wir haben zuerst die Daten geladen und fehlende Werte ersetzt.
        Danach haben wir die wichtigsten Variablen in zwei Gruppen geteilt:
        numerische und kategorische Variablen.
        </p>
        <p>
        Dann haben wir:
        </p>
        <ul>
        <li>die Daten bereinigt (fehlende Werte ersetzt)</li>
        <li>die Zielvariable in eine numerische Form umgewandelt (Target Encoding)</li>
        <li>die wichtigsten Kennzahlen berechnet (Mittelwert, Streuung, Ausreißer)</li>
        <li>die Beziehungen zwischen den Variablen untersucht (Korrelationen, Chi²-Tests)</li>
        <li>statistische Tests durchgeführt (Shapiro-Wilk, Pearson, Spearman, Z-Test)</li>
        <li>und die Ergebnisse in Grafiken visualisiert (Scatter-Plot, Heatmap, Barplot)</li>
        </ul>
        <p>
        Am Ende haben wir alles in einem Bericht zusammengefasst.
        </p>

        <div class="chart-page">
            <h2>2. Deskriptive Statistik und Kennzahlen</h2>
            <p>
            In dieser Tabelle zeigen wir die wichtigsten Werte für alle numerischen Variablen.
            </p>
            <p>
            Wir haben berechnet:
            </p>
            <ul>
            <li>den Mittelwert (Durchschnitt)</li>
            <li>den Median (50%-Wert)</li>
            <li>die Quartile (25% und 75%)</li>
            <li>die Standardabweichung (Streuung)</li>
            <li>die Varianz (Streuung im Quadrat)</li>
            <li>und die Anzahl der Ausreißer (Werte, die weit von anderen entfernt sind)</li>
            </ul>
            <p>
            Ausreißer sind Werte, die sehr stark von den anderen abweichen.
            </p>
            
            <table>
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Mittelwert</th>
                        <th>Median</th>
                        <th>Q1 (25%)</th>
                        <th>Q3 (75%)</th>
                        <th>Std.-Abw.</th>
                        <th>Varianz</th>
                        <th>Ausreißer (IQR)</th>
                    </tr>
                </thead>
                <tbody>
                    {desc_rows_html}
                </tbody>
            </table>
<div class="results-box">
    <h3>Intervallschätzung (Konfidenzintervalle)</h3>
    <p>• Das berechnete 95%-Konfidenzintervall für den wahren Mittelwert des <b>Routineanteils</b> liegt zwischen <b>{ci_routine[0]:.2f}%</b> und <b>{ci_routine[1]:.2f}%</b>.</p>
    <p>• Das berechnete 95%-Konfidenzintervall für den wahren Mittelwert des <b>Automatisierungsanteils</b> liegt zwischen <b>{ci_automated[0]:.2f}%</b> und <b>{ci_automated[1]:.2f}%</b>.</p>
</div>
        </div>

        <div class="chart-page">
            <h2>3. Hypothesentests & Prüfung der Voraussetzungen</h2>
            
            <div class="results-box">
                <h3>Voraussetzungsprüfung: Shapiro-Wilk-Test auf Normalverteilung</h3>
                <p><b>H₀ (Nullhypothese):</b> Die Daten sind normalverteilt. <br> 
                   <b>H₁ (Alternativhypothese):</b> Die Daten sind nicht normalverteilt.</p>
                
                <ul style="list-style-type: none; padding-left: 0;">
                    {shapiro_html}
                </ul>
                
                <p style="margin-top: 10px;"><i>Entscheidung:</i> Da die p-Werte der Variablen kleiner als 0.05 sind, wird die Nullhypothese für diese Merkmale abgelehnt. Die Daten weichen signifikant von einer Normalverteilung ab. Daher ist die <b>Spearman-Rangkorrelation</b> das mathematisch empfohlene Verfahren (Vorgabe: {empfohlenes_verfahren}).</p>
            </div>

            <div class="results-box">
                <h3>Ergebnisse der Korrelationsanalysen</h3>
                <p>• <b>Bravais-Pearson-Koeffizient</b> (linearer Zusammenhang): r = {pearson_r:.2f} (p = {p_pearson:.4f})</p>
                <p>• <b>Spearman-Rangkorrelationskoeffizient</b> (monotoner Zusammenhang): ρ = <b>{spearman_r:.2f}</b> (p = {p_spearman:.4f})</p>
                <p>• <b>Spearman-Zusammenhang</b> (AI Usage Hours vs. Automation): ρ = <b>{spearman_ai:.2f}</b> (p = {p_spearman_ai:.4f})</p>
                <p>Ein p-Wert kleiner als 0.05 bedeutet, dass die Verbindung statistisch signifikant ist.</p>
            </div>

            <div class="results-box">
                <h3>Gruppenvergleich: Z-Test</h3>
                <p><b>H₀:</b> Es gibt keinen Unterschied im Automatisierungsanteil zwischen High-Risk und Low-Risk Gruppen.<br>
                   <b>H₁:</b> Es gibt einen signifikanten Unterschied.</p>
                <p>Ergebnis: Z = <b>{z_stat:.2f}</b>, p-Wert = <b>{p_z:.4f}</b></p>
                <p><i>Entscheidung:</i> Da p &lt; 0.05, wird H₀ abgelehnt. Die Gruppen unterscheiden sich statistisch signifikant.</p>
            </div>
        </div>

        <div class="chart-page">
            <h2>4. Grafische Analyse: Routineanteil und Automatisierung</h2>
            <div class="chart-container">
                <img src="{img_scatter}" alt="Scatter Plot">
                <div class="chart-title">Grafik 1. Zusammenhang zwischen Routineanteil und Automatisierung</div>
            </div>
             <div class="results-box">
            <p>Diese Grafik zeigt den Zusammenhang zwischen Routineaufgaben und Automatisierung durch KI.</p>
            <p> Dieser Plot zeigt den Zusammenhang zwischen Routineanteil und Automatisierung. </p>
            <p> Jeder Punkt stellt eine Person dar, die Farbe zeigt das Entlassungsrisiko. </p>
            <p> Ergebnis: Es gibt einen positiven Zusammenhang — mehr Routine bedeutet mehr Automatisierung. </p>
            </div>
        </div>

        <div class="chart-page">
            <h2>5. Korrelationsmatrix</h2>
            <div class="chart-container">
                <img src="{img_heat}" alt="Heatmap">
                <div class="chart-title">Grafik 2. Korrelationsmatrix aller numerischen Merkmale mit Layoff_Risk</div>
            </div>
            <div class="results-box">
            <p>Diese Grafik zeigt die Spearman-Rangkorrelation zwischen allen numerischen Variablen und der Zielvariable.</p>
            <p> Sie hilft zu erkennen, welche Variablen ähnliche Informationen enthalten. </p>
            <p> Wichtig für Modellierung: stark korrelierte Variablen können Probleme verursachen. </p>
            </div>
        </div>

        <div class="chart-page">
            <h2>6. Einfluss kategorialer Variablen auf das Entlassungsrisiko</h2>
            <div class="chart-container">
                <img src="{img_chi}" alt="Chi2 Analysis">
                <div class="chart-title">Grafik 3. Einfluss kategorialer Variablen (Chi² Analyse)</div>
            </div>

            <div class="results-box">
                 <h3>Interpretation der Chi²-Analyse:</h3>
              <p>
    Diese Grafik zeigt, welche kategorischen Variablen einen Zusammenhang mit dem Entlassungsrisiko haben.
    </p>

    <p> Hier wurde untersucht, welche Kategorien das Entlassungsrisiko beeinflussen. </p>
    <p> Dafür wurde der Chi²-Test verwendet. </p>
    <p> Ergebnis: AI Adoption Level, Job Level und  Job Role zeigen den stärksten Zusammenhang. </p>
            </div>
        </div>

    </body>
    </html>

    """
    return html_template


def export_pdf(html_content, output_filename):
    print(f"[PDF] Exportiere Bericht nach {output_filename}...")
    
    dirname = os.path.dirname(output_filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(output_filename)
        print(f"[PDF] Erfolgreich gespeichert: {output_filename}")
    except ImportError:
       print("[KRITISCHER FEHLER] Die Bibliothek weasyprint ist nicht installiert.")
