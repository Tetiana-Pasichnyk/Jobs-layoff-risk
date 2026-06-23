import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def fig_to_base64(fig_obj, filename=None):
    if filename:
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            
        fig_obj.savefig(filename, format='png', bbox_inches='tight', dpi=150)
        print(f"[PLOTS] Grafik erfolgreich gespeichert: {filename}")
    
    buf = BytesIO()
    fig_obj.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig_obj)
    return f"data:image/png;base64,{img_str}"


def create_plots(df, chi_df, numerische_spalten, output_dir="output_analysis"):

    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "grid.color": "#E5E5E5"
    })

    color_hellgruen = '#AFBA12'        # Low
    color_dunkelorange = '#CA5902'     # Medium
    color_red = '#d62728'              # High
    color_dunkelgruen = '#175C00'      # Additional

    # ==========================================================================
    # 1. Scatter Plot
    # ==========================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    df_sample = df.sample(min(len(df), 1000), random_state=42)

    sns.scatterplot(
        data=df_sample,
        x='Routine_Task_Percentage',
        y='Tasks_Automated_Percentage',
        hue='Layoff_Risk',
        palette={
            'Low': color_dunkelgruen,
            'Medium': color_dunkelorange,
            'High': color_red
        },
        alpha=0.7,
        ax=ax1
    )

    ax1.set_title(
        "Zusammenhang zwischen Routineanteil und Automatisierung",
        fontsize=11,
        fontweight='bold',
        color='#1e3d59'
    )
    plt.tight_layout()
    
    scatter_path = os.path.join(output_dir, "scatter_plot.png")
    img_scatter = fig_to_base64(fig1, filename=scatter_path)

    # ==========================================================================
    # 2. Heatmap (Spearman-Korrelation)
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    corr_columns = numerische_spalten + ['Layoff_Risk_Numeric']
    corr_matrix = df[corr_columns].corr(method='spearman')

    colors = [color_dunkelgruen, color_hellgruen, '#d3d3d3', color_dunkelorange, color_red]
    custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors)

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=custom_cmap,
        center=0,
        vmin=-1, vmax=1,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=ax2,
        cbar=True,
        annot_kws={"size": 8}
    )

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=9)

    ax2.set_title(
        "Korrelationsmatrix (Spearman) mit Zielvariable",
        fontsize=11,
        fontweight='bold',
        color='#1e3d59'
    )
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    img_heat = fig_to_base64(fig2, filename=heatmap_path)

    # ==========================================================================
    # 3. Chi²-Barplot
    # ==========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))

    custom_colors = []
    for feature in chi_df["Feature"]:
        if feature == "AI_Adoption_Level":
            custom_colors.append(color_red)
        elif feature in ["Job_Level", "Job_Role"]:
            custom_colors.append(color_dunkelorange)
        elif feature in ["Education_Level", "Industry"]:
            custom_colors.append(color_hellgruen)
        else:
            custom_colors.append(color_dunkelgruen)

    sns.barplot(
        data=chi_df,
        x="Chi2",
        y="Feature",
        palette=custom_colors,
        ax=ax3
    )

    ax3.set_title(
        "Einfluss kategorialer Variablen (Chi²-Analyse)",
        fontsize=11,
        fontweight='bold',
        color='#1e3d59'
    )
    ax3.set_xlabel("Chi²-Wert", fontsize=10)
    ax3.set_ylabel("Merkmal", fontsize=10)
    plt.tight_layout()
    

    chi_path = os.path.join(output_dir, "chi2_barplot.png")
    img_chi = fig_to_base64(fig3, filename=chi_path)

    print(f"[PLOTS] Alle Visualisierungen wurden im Ordner gespeichert: {output_dir}")

    return {
        "scatter": img_scatter,
        "heatmap": img_heat,
        "chi": img_chi
    }