import os

def build_html(
    desc_stats,
    outliers_dict,
    ci_routine,
    ci_automated,
    shapiro_html,
    pearson_r,
    p_pearson,
    spearman_r,
    p_spearman,
    z_tests,
    img_scatter,
    img_heat,
    img_chi,
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
            font-family: Arial, sans-serif;
            color: #333;
            font-size: 11pt;
            line-height: 1.5;
        }}
        .header {{
            background-color: #d4e84e;
            padding: 20px;
            border-radius: 5px;
        }}
        h2 {{
            border-left: 4px solid #ffc107;
            padding-left: 8px;
        }}
        .results-box {{
            background: #faf7f4;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6px;
        }}
        th {{
            background: #d4e84e;
        }}
        .chart-page {{
            page-break-before: always;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            max-width: 85%;
        }}
    </style>
    </head>

    <body>

    <div class="header">
        <h1>Analysebericht: Einfluss von KI auf Arbeitsplätze</h1>
    </div>

    <h2>1. Deskriptive Statistik</h2>

    <table>
        <thead>
            <tr>
                <th>Variable</th>
                <th>Mittelwert</th>
                <th>Median</th>
                <th>Q1</th>
                <th>Q3</th>
                <th>Std</th>
                <th>Varianz</th>
                <th>Ausreißer</th>
            </tr>
        </thead>
        <tbody>
            {desc_rows_html}
        </tbody>
    </table>

    <div class="results-box">
        <h3>Konfidenzintervalle</h3>
        <p>Routine: {ci_routine[0]:.2f}% – {ci_routine[1]:.2f}%</p>
        <p>Automation: {ci_automated[0]:.2f}% – {ci_automated[1]:.2f}%</p>
    </div>

    <div class="results-box">
        <h3>Shapiro-Wilk-Test</h3>
        <ul>
            {shapiro_html}
        </ul>

        <p>
        Da die Daten nicht normalverteilt sind, wurde Spearman verwendet.
        </p>
    </div>

    <div class="results-box">
        <h3>Korrelationsanalyse</h3>
        <p>Pearson: r = {pearson_r:.2f} (p = {p_pearson:.4f})</p>
        <p>Spearman: ρ = {spearman_r:.2f} (p = {p_spearman:.4f})</p>
    </div>

    <div class="results-box">
        <h3>Z-Tests (Gruppenvergleich)</h3>
        <p><b>H₀:</b> Kein Unterschied zwischen High- und Low-Risk Gruppen</p>
        <p><b>H₁:</b> Unterschied besteht</p>

        <pre>
{z_tests}
        </pre>

        <p>p &lt; 0.05 → signifikante Unterschiede</p>
    </div>

    <div class="chart-page">
        <h2>Visualisierungen</h2>

        <div class="chart-container">
            <img src="{img_scatter}">
            <p>Routine vs Automation</p>
        </div>

        <div class="chart-container">
            <img src="{img_heat}">
            <p>Korrelationsmatrix</p>
        </div>

        <div class="chart-container">
            <img src="{img_chi}">
            <p>Chi² Analyse</p>
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
        print("[FEHLER] weasyprint nicht installiert")