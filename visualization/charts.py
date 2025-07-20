"""
visualization/charts.py — Professionelle Business-Charts & Dashboard.

Design-Philosophie: Kein "Default Matplotlib". Jedes Chart folgt
den Prinzipien professioneller Daten-Kommunikation:
- Klare Aussage (Chart-Titel erklärt den Insight, nicht nur "was")
- Dezente Gestaltung (weniger ist mehr, kein Chartjunk)
- Konsistente Farbpalette (Corporate Blue)
- Alle Labels & Achsen beschriftet
- Quelle/Zeitraum immer sichtbar

Referenz-Stil: Financial Times, McKinsey-Charts, Economist Graphics.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend für headless Execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import VisualizationConfig, AnalysisConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────
# GLOBAL STYLE SETUP
# ─────────────────────────────────────────────

def setup_style(config: VisualizationConfig) -> None:
    """
    Setzt globales Matplotlib-Styling für konsistente Chart-Ästhetik.

    Ziel: Charts sehen sofort "professionell" aus, nicht wie
    Notebook-Schnellschuesse. Wird einmal am Pipeline-Start aufgerufen.
    """
    colors = config.colors

    # Custom Style-Parameter
    plt.rcParams.update({
        # Fonts
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,

        # Axes
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  True,
        "axes.spines.bottom": True,
        "axes.linewidth":    0.8,
        "axes.edgecolor":    "#CCCCCC",
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",

        # Grid (dezent)
        "axes.grid":         True,
        "grid.color":        "#E8E8E8",
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",
        "axes.axisbelow":    True,

        # Lines
        "lines.linewidth":   2.0,
        "lines.markersize":  6,

        # Saving
        "savefig.dpi":       config.dpi,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

    logger.info("Chart-Styling konfiguriert")


def _add_chart_footer(ax, source: str = "RetailCo Sales DB", config: Optional[VisualizationConfig] = None) -> None:
    """Fügt Quelle + Generierungsdatum als Fußzeile ein."""
    date_str = datetime.now().strftime("%d.%m.%Y")
    ax.annotate(
        f"Quelle: {source}  |  Erstellt: {date_str}",
        xy=(1, -0.12), xycoords="axes fraction",
        ha="right", va="bottom",
        fontsize=7, color="#999999",
        style="italic"
    )


def _format_euro(val: float, pos=None) -> str:
    """Axis-Formatter: Euro mit Tausendertrennzeichen."""
    if abs(val) >= 1_000_000:
        return f"€{val/1_000_000:.1f}M"
    elif abs(val) >= 1_000:
        return f"€{val/1_000:.0f}K"
    return f"€{val:.0f}"


# ─────────────────────────────────────────────
# CHART 1: EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────

def create_executive_dashboard(
    monthly_df: pd.DataFrame,
    kpis: dict,
    config: VisualizationConfig,
    output_dir: Path
) -> Path:
    """
    Executive Dashboard: Überblick auf einer Seite.

    Layout: 2×3 Grid
    - Row 1: 4 KPI-Karten (Revenue, Growth, Margin, Orders)
    - Row 2: Monatlicher Umsatzverlauf mit Trend-Linie

    Zielgruppe: C-Level, die 30 Sekunden für einen Chart haben.

    Args:
        monthly_df: Monatliche Umsätze mit Wachstumsraten
        kpis: KPI-Dictionary aus kpi_calculator
        config: Visualisierungs-Konfiguration
        output_dir: Ausgabe-Pfad

    Returns:
        Pfad zur gespeicherten PNG-Datei
    """
    colors = config.colors

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "Executive Sales Dashboard",
        fontsize=18, fontweight="bold", y=0.98,
        color=colors["primary"]
    )

    # ── Gridspec: KPI-Karten oben, Zeitreihe unten ────────────
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[0.8, 2.5],
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.96, top=0.92, bottom=0.08
    )

    # ── KPI-Karten (4 Stück) ──────────────────────────────────
    rev = kpis["revenue"]
    growth = kpis["growth"]
    customers = kpis["customers"]

    kpi_cards = [
        {
            "title": "TOTAL REVENUE",
            "value": f"€{rev['total']/1e6:.2f}M",
            "sub": f"{'+' if growth['yoy_pct'] and growth['yoy_pct'] > 0 else ''}{growth['yoy_pct']:.1f}% YoY" if growth["yoy_pct"] else "",
            "color": colors["primary"],
            "positive": (growth["yoy_pct"] or 0) > 0
        },
        {
            "title": "Ø MOM GROWTH",
            "value": f"{growth['avg_mom_pct_last6m']:+.1f}%" if growth["avg_mom_pct_last6m"] else "N/A",
            "sub": "Letzte 6 Monate",
            "color": colors["secondary"],
            "positive": (growth["avg_mom_pct_last6m"] or 0) > 0
        },
        {
            "title": "GROSS MARGIN",
            "value": f"{rev['gross_margin_pct']:.1f}%" if rev["gross_margin_pct"] else "N/A",
            "sub": "Gewichtete Ø-Marge",
            "color": colors["accent"],
            "positive": True
        },
        {
            "title": "ACTIVE CUSTOMERS",
            "value": f"{customers['active_total']:,}".replace(",", "."),
            "sub": f"{customers['champions_pct']:.1f}% Champions",
            "color": colors["positive"],
            "positive": True
        },
    ]

    for i, card in enumerate(kpi_cards):
        ax_card = fig.add_subplot(gs[0, i])
        ax_card.set_xlim(0, 1)
        ax_card.set_ylim(0, 1)
        ax_card.axis("off")

        # Karten-Hintergrund
        rect = mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.02",
            facecolor=colors["light"],
            edgecolor=card["color"],
            linewidth=2
        )
        ax_card.add_patch(rect)

        # KPI-Titel
        ax_card.text(0.5, 0.82, card["title"],
                     ha="center", va="center", fontsize=8,
                     color=colors["neutral"], fontweight="bold",
                     transform=ax_card.transAxes)

        # KPI-Wert (groß)
        value_color = colors["positive"] if card["positive"] else colors["negative"]
        ax_card.text(0.5, 0.52, card["value"],
                     ha="center", va="center", fontsize=16,
                     fontweight="bold", color=card["color"],
                     transform=ax_card.transAxes)

        # Untertitel
        ax_card.text(0.5, 0.22, card["sub"],
                     ha="center", va="center", fontsize=8,
                     color=colors["neutral"],
                     transform=ax_card.transAxes)

    # ── Monatlicher Umsatzverlauf (Hauptchart) ─────────────────
    ax_main = fig.add_subplot(gs[1, :])

    df = monthly_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    x = range(len(df))
    months = df["month"].dt.strftime("%b %Y")

    # Actual Revenue als Balken
    bars = ax_main.bar(
        x, df["revenue"],
        color=colors["accent"], alpha=0.6,
        label="Monatsumsatz", zorder=2
    )

    # Q4-Monate hervorheben
    q4_mask = df["month"].dt.month.isin([10, 11, 12])
    for idx, (bar, is_q4) in enumerate(zip(bars, q4_mask)):
        if is_q4:
            bar.set_color(colors["primary"])
            bar.set_alpha(0.85)

    # Moving Average Linien
    ax_main.plot(
        x, df["revenue_ma3"],
        color=colors["secondary"], linewidth=2.5,
        label="3-Monats-Trend", zorder=3
    )
    ax_main.plot(
        x, df["revenue_ma6"],
        color=colors["negative"], linewidth=1.5,
        linestyle="--", label="6-Monats-Trend", zorder=3
    )

    # Lineare Trendlinie
    z = np.polyfit(range(len(df)), df["revenue"], 1)
    p = np.poly1d(z)
    ax_main.plot(
        x, p(range(len(df))),
        color="#888888", linewidth=1, linestyle=":",
        label=f"Linearer Trend", alpha=0.8, zorder=3
    )

    # Black Friday Annotation
    bf_idx = df[
        (df["month"].dt.month == 11) &
        (df["revenue"] == df[df["month"].dt.month == 11]["revenue"].max())
    ].index
    if len(bf_idx) > 0:
        bf_pos = df.index.get_loc(bf_idx[0])
        ax_main.annotate(
            "Black Friday",
            xy=(bf_pos, df.iloc[bf_pos]["revenue"]),
            xytext=(bf_pos + 1.5, df.iloc[bf_pos]["revenue"] * 1.08),
            fontsize=8, color=colors["negative"],
            arrowprops=dict(arrowstyle="->", color=colors["negative"], lw=1),
        )

    # Achsen formatieren
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(_format_euro))
    ax_main.set_xticks(x[::2])
    ax_main.set_xticklabels(months.iloc[::2], rotation=45, ha="right", fontsize=8)
    ax_main.set_title(
        "Monatlicher Umsatzverlauf  |  Q4-Monate hervorgehoben  |  Trend-Linien eingeblendet",
        fontsize=11, pad=12
    )
    ax_main.set_ylabel("Umsatz")
    ax_main.legend(loc="upper left", framealpha=0.9)

    # Q4-Label
    ax_main.text(
        0.99, 0.96, "■ Q4-Monate",
        transform=ax_main.transAxes,
        ha="right", va="top", fontsize=8,
        color=colors["primary"], style="italic"
    )

    _add_chart_footer(ax_main)

    output_path = output_dir / "chart_01_executive_dashboard.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart 1 gespeichert: {output_path.name}")
    return output_path


# ─────────────────────────────────────────────
# CHART 2: PRODUKT-PERFORMANCE MATRIX (BCG/Bubble)
# ─────────────────────────────────────────────

def create_product_matrix(
    products_df: pd.DataFrame,
    config: VisualizationConfig,
    output_dir: Path
) -> Path:
    """
    BCG-inspirierte Produkt-Performance Matrix als Bubble Chart.

    X-Achse:    Gesamtumsatz 2024
    Y-Achse:    Wachstumsrate H1→H2
    Bubble-Größe: Bruttomarge in %
    Farbe:      Produktkategorie

    Insight sofort sichtbar: Welche Produkte sind Stars, welche Dogs?

    Args:
        products_df: Ergebnis von compute_product_growth_matrix() +
                     compute_abc_analysis()
        config: Visualisierungs-Konfiguration
        output_dir: Ausgabe-Pfad

    Returns:
        Pfad zur gespeicherten PNG-Datei
    """
    colors = config.colors
    cat_colors = colors["categories"]

    fig, ax = plt.subplots(figsize=(14, 9))

    # Kategorien & Farben mappen
    categories = products_df["category"].unique()
    cat_color_map = {cat: cat_colors[i % len(cat_colors)] for i, cat in enumerate(categories)}

    # Bubble Chart
    for _, row in products_df.iterrows():
        if pd.isna(row.get("total_revenue_2024")) or pd.isna(row.get("growth_h1_to_h2")):
            continue

        bubble_size = max(50, row.get("margin_pct", 30) * 15)
        cat = row.get("category", "Other")

        ax.scatter(
            row["total_revenue_2024"],
            row["growth_h1_to_h2"],
            s=bubble_size,
            c=cat_color_map.get(cat, "#999999"),
            alpha=0.75,
            edgecolors="white",
            linewidth=1.5,
            zorder=3
        )

        # Label für Top-Produkte
        if row["total_revenue_2024"] > products_df["total_revenue_2024"].quantile(0.5):
            short_name = row["name"][:18] + "…" if len(str(row["name"])) > 18 else row["name"]
            ax.annotate(
                short_name,
                (row["total_revenue_2024"], row["growth_h1_to_h2"]),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7.5,
                color="#444444"
            )

    # Quadranten-Linien (Median-basiert)
    rev_median = products_df["total_revenue_2024"].median()
    growth_median = products_df["growth_h1_to_h2"].median()

    ax.axvline(rev_median, color="#CCCCCC", linewidth=1, linestyle="--", zorder=1)
    ax.axhline(growth_median, color="#CCCCCC", linewidth=1, linestyle="--", zorder=1)
    ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)

    # Quadranten-Labels
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    pad_x = (ax_xlim[1] - ax_xlim[0]) * 0.03
    pad_y = (ax_ylim[1] - ax_ylim[0]) * 0.03

    quadrant_labels = [
        (ax_xlim[0] + pad_x, ax_ylim[1] - pad_y, "QUESTION MARKS\n(Invest or Exit?)", "left", "top"),
        (ax_xlim[1] - pad_x, ax_ylim[1] - pad_y, "** STARS **\n(Invest & Grow)", "right", "top"),
        (ax_xlim[0] + pad_x, ax_ylim[0] + pad_y, "DOGS\n(Review & Rationalize)", "left", "bottom"),
        (ax_xlim[1] - pad_x, ax_ylim[0] + pad_y, "CASH COWS\n(Harvest & Maintain)", "right", "bottom"),
    ]

    for qx, qy, qlabel, ha, va in quadrant_labels:
        ax.text(qx, qy, qlabel,
                ha=ha, va=va, fontsize=8,
                color="#AAAAAA", style="italic", alpha=0.8)

    # Legende Kategorien
    legend_handles = [
        mpatches.Patch(color=cat_color_map[cat], label=cat)
        for cat in categories
    ]
    ax.legend(handles=legend_handles, title="Kategorie",
              loc="center right", framealpha=0.9, fontsize=8)

    # Bubble-Größen-Hinweis
    ax.text(0.01, 0.01,
            "Bubble-Größe = Bruttomarge %",
            transform=ax.transAxes,
            fontsize=7.5, color=colors["neutral"], style="italic")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_euro))
    ax.set_xlabel("Gesamtumsatz 2024")
    ax.set_ylabel("Wachstum H1 → H2 (%)")
    ax.set_title(
        "Produkt-Performance Matrix  |  BCG-Quadranten  |  Bubble = Marge",
        fontsize=12, pad=15
    )

    _add_chart_footer(ax)

    output_path = output_dir / "chart_02_product_matrix.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart 2 gespeichert: {output_path.name}")
    return output_path


# ─────────────────────────────────────────────
# CHART 3: REGIONALE HEATMAP
# ─────────────────────────────────────────────

def create_regional_heatmap(
    region_month_df: pd.DataFrame,
    config: VisualizationConfig,
    output_dir: Path
) -> Path:
    """
    Seaborn-Heatmap: Monat × Region → Umsatz als Farbintensität.

    Dieser Chart-Typ ist in Consulting-Decks universell:
    zeigt sofort saisonale Muster JE Region (Performance-Shift sichtbar).

    Args:
        region_month_df: DataFrame mit month, region, revenue
        config: Visualisierungs-Konfiguration
        output_dir: Ausgabe-Pfad

    Returns:
        Pfad zur gespeicherten PNG-Datei
    """
    colors = config.colors

    # Pivot für Heatmap: Monate als Zeilen, Regionen als Spalten
    df = region_month_df.copy()
    df["month"] = pd.to_datetime(df["month"])

    pivot = df.pivot_table(
        index="month", columns="region",
        values="revenue", aggfunc="sum"
    )

    # Monat-Labels kompakt
    month_labels = pivot.index.strftime("%b %y")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Seaborn Heatmap mit Annotationen (Werte sichtbar)
    sns.heatmap(
        pivot / 1000,  # in Tausend für lesbare Annotationen
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="#f0f0f0",
        ax=ax,
        cbar_kws={"label": "Umsatz in T€", "shrink": 0.8},
        annot_kws={"size": 8.5}
    )

    ax.set_yticklabels(month_labels, rotation=0, fontsize=8.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_xlabel("Region", fontsize=10)
    ax.set_ylabel("Monat")
    ax.set_title(
        "Regionale Umsatz-Heatmap  |  Saisonalität & regionale Stärken sichtbar  |  Werte in T€",
        fontsize=12, pad=15
    )

    # Q4-Zeilen hervorheben (Monate 10, 11, 12)
    for i, month in enumerate(pivot.index):
        if month.month in [10, 11, 12]:
            ax.add_patch(
                mpatches.Rectangle(
                    (0, i), len(pivot.columns), 1,
                    fill=False, edgecolor=colors["primary"],
                    linewidth=2, clip_on=False, zorder=5
                )
            )

    _add_chart_footer(ax)

    output_path = output_dir / "chart_03_regional_heatmap.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart 3 gespeichert: {output_path.name}")
    return output_path


# ─────────────────────────────────────────────
# CHART 4: RFM KUNDEN-SEGMENTIERUNG
# ─────────────────────────────────────────────

def create_rfm_segmentation(
    rfm_df: pd.DataFrame,
    config: VisualizationConfig,
    output_dir: Path
) -> Path:
    """
    RFM-Segmentierungs-Chart: Scatter + Segment-Verteilung.

    Layout: Links Scatter (Recency vs. Monetary, Größe=Frequency),
            Rechts Horizontaler Balken der Segment-Verteilung.

    Args:
        rfm_df: Ergebnis von calculate_rfm()
        config: Visualisierungs-Konfiguration
        output_dir: Ausgabe-Pfad

    Returns:
        Pfad zur gespeicherten PNG-Datei
    """
    colors = config.colors
    cat_colors = colors["categories"]

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(16, 8))

    # Segmente & Farben
    segments = rfm_df["rfm_segment"].unique()
    seg_color_map = {seg: cat_colors[i % len(cat_colors)] for i, seg in enumerate(segments)}

    # ── Scatter: Recency vs. Monetary ─────────────────────────
    for seg in segments:
        mask = rfm_df["rfm_segment"] == seg
        sub = rfm_df[mask]
        ax_scatter.scatter(
            sub["recency"],
            sub["monetary"],
            s=sub["frequency"] * 8,
            c=seg_color_map[seg],
            alpha=0.55,
            edgecolors="white",
            linewidth=0.5,
            label=seg,
            zorder=3
        )

    ax_scatter.set_xlabel("Recency (Tage seit letztem Kauf) — niedriger = besser →")
    ax_scatter.set_ylabel("Monetary (Gesamtumsatz €)")
    ax_scatter.set_title("RFM-Segmente: Recency vs. Monetary\nBubble-Größe = Kaufhäufigkeit")
    ax_scatter.yaxis.set_major_formatter(mticker.FuncFormatter(_format_euro))
    ax_scatter.legend(loc="upper right", fontsize=8, title="Segment")
    ax_scatter.invert_xaxis()  # Links = aktuell (besser), rechts = inaktiv

    # Anmerkung: warum X invertiert
    ax_scatter.text(0.99, 0.01,
                    "← Aktive Kunden  |  Inaktive Kunden →",
                    transform=ax_scatter.transAxes,
                    ha="right", va="bottom", fontsize=7.5,
                    color=colors["neutral"], style="italic")

    _add_chart_footer(ax_scatter)

    # ── Balken: Segment-Verteilung ─────────────────────────────
    seg_counts = rfm_df["rfm_segment"].value_counts().sort_values()
    seg_revenue = rfm_df.groupby("rfm_segment")["monetary"].sum().reindex(seg_counts.index)

    bar_colors = [seg_color_map.get(s, "#999999") for s in seg_counts.index]

    bars = ax_bar.barh(
        seg_counts.index, seg_counts.values,
        color=bar_colors, alpha=0.85,
        edgecolor="white", linewidth=0.5
    )

    # Wert-Labels
    for bar, count in zip(bars, seg_counts.values):
        ax_bar.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{count} ({count/len(rfm_df)*100:.0f}%)",
            va="center", fontsize=8.5, color="#444444"
        )

    ax_bar.set_xlabel("Anzahl Kunden")
    ax_bar.set_title("Kundenverteilung nach RFM-Segment\nFür gezielte Marketingmaßnahmen")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    _add_chart_footer(ax_bar)

    plt.tight_layout(pad=3)

    output_path = output_dir / "chart_04_rfm_segmentation.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart 4 gespeichert: {output_path.name}")
    return output_path


# ─────────────────────────────────────────────
# CHART 5: TREND-ANALYSE
# ─────────────────────────────────────────────

def create_trend_analysis(
    monthly_df: pd.DataFrame,
    config: VisualizationConfig,
    output_dir: Path
) -> Path:
    """
    Mehrteilige Trend-Analyse: Actual, Trend, Seasonal, YoY.

    Layout: 2×2 Subplot
    - Oben links:  Actual + Moving Average + Forecast
    - Oben rechts: MoM-Wachstumsraten (Balken)
    - Unten links: YoY-Vergleich 2023 vs 2024
    - Unten rechts: Quartalsweise kumulierte Umsätze

    Args:
        monthly_df: Monatliche Umsätze mit Wachstumsraten
        config: Visualisierungs-Konfiguration
        output_dir: Ausgabe-Pfad

    Returns:
        Pfad zur gespeicherten PNG-Datei
    """
    colors = config.colors
    df = monthly_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Trend-Analyse & Wachstumsdynamik", fontsize=15,
                 fontweight="bold", y=1.01)

    # ── (0,0) Actual + Trend + Forecast ───────────────────────
    ax = axes[0, 0]
    x = range(len(df))

    ax.fill_between(x, df["revenue"], alpha=0.2, color=colors["accent"])
    ax.plot(x, df["revenue"], color=colors["accent"], linewidth=1.5,
            label="Tatsächlich", alpha=0.8)
    ax.plot(x, df["revenue_ma6"], color=colors["primary"], linewidth=2.5,
            label="6M Gleitender Ø")

    # Einfacher linearer Forecast (3 Monate)
    n = len(df)
    z = np.polyfit(range(n), df["revenue"], 1)
    p = np.poly1d(z)
    forecast_x = range(n - 1, n + 3)
    ax.plot(forecast_x, p(forecast_x),
            color=colors["negative"], linewidth=2, linestyle="--",
            label="Forecast (linear)", zorder=4)
    ax.axvspan(n - 1, n + 2, alpha=0.07, color=colors["negative"])

    ax.set_xticks(list(x)[::3])
    ax.set_xticklabels(df["month"].dt.strftime("%b %y").iloc[::3], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_euro))
    ax.set_title("Umsatzverlauf mit Trendlinie & Forecast")
    ax.legend(fontsize=8)
    _add_chart_footer(ax)

    # ── (0,1) MoM-Wachstumsraten ──────────────────────────────
    ax = axes[0, 1]
    mom = df.dropna(subset=["mom_growth_pct"])
    bar_colors = [colors["positive"] if v >= 0 else colors["negative"]
                  for v in mom["mom_growth_pct"]]

    ax.bar(range(len(mom)), mom["mom_growth_pct"],
           color=bar_colors, alpha=0.8, width=0.7)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axhline(mom["mom_growth_pct"].mean(), color=colors["secondary"],
               linewidth=1.5, linestyle="--",
               label=f"Ø MoM: {mom['mom_growth_pct'].mean():.1f}%")

    ax.set_xticks(range(0, len(mom), 2))
    ax.set_xticklabels(mom["month"].dt.strftime("%b %y").iloc[::2], rotation=45, ha="right")
    ax.set_title("Month-over-Month Wachstumsraten (%)")
    ax.set_ylabel("Wachstum (%)")
    ax.legend(fontsize=8)
    _add_chart_footer(ax)

    # ── (1,0) YoY-Vergleich: 2023 vs 2024 ─────────────────────
    ax = axes[1, 0]
    df_2023 = df[df["month"].dt.year == 2023].copy()
    df_2024 = df[df["month"].dt.year == 2024].copy()

    if len(df_2023) > 0 and len(df_2024) > 0:
        x_2023 = df_2023["month"].dt.month - 1
        x_2024 = df_2024["month"].dt.month - 1

        ax.plot(x_2023, df_2023["revenue"], color=colors["neutral"],
                linewidth=2, marker="o", markersize=5, label="2023", linestyle="--")
        ax.plot(x_2024, df_2024["revenue"], color=colors["primary"],
                linewidth=2.5, marker="o", markersize=5, label="2024")

        # Füllbereich zwischen Jahren
        # Nur überlappende Monate
        common_months = set(x_2023) & set(x_2024)
        if common_months:
            shared_x = sorted(common_months)
            rev_23 = df_2023.set_index(df_2023["month"].dt.month - 1)["revenue"].reindex(shared_x)
            rev_24 = df_2024.set_index(df_2024["month"].dt.month - 1)["revenue"].reindex(shared_x)
            ax.fill_between(shared_x, rev_23, rev_24,
                            where=rev_24 >= rev_23,
                            alpha=0.15, color=colors["positive"], label="Wachstum YoY")

        month_names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                       "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names, fontsize=8.5)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_euro))
    ax.set_title("Year-over-Year Vergleich: 2023 vs. 2024")
    ax.legend(fontsize=8)
    _add_chart_footer(ax)

    # ── (1,1) Quartals-Kumulation ──────────────────────────────
    ax = axes[1, 1]
    quarterly = df.groupby("quarter")["revenue"].sum().reset_index()
    quarterly = quarterly.sort_values("quarter")

    bar_colors_q = []
    for q in quarterly["quarter"]:
        if "Q4" in q:
            bar_colors_q.append(colors["primary"])
        elif "Q3" in q:
            bar_colors_q.append(colors["secondary"])
        else:
            bar_colors_q.append(colors["accent"])

    bars = ax.bar(
        range(len(quarterly)), quarterly["revenue"],
        color=bar_colors_q, alpha=0.85,
        edgecolor="white", linewidth=0.5
    )

    # Wert-Labels auf Balken
    for bar, val in zip(bars, quarterly["revenue"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            _format_euro(val),
            ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

    ax.set_xticks(range(len(quarterly)))
    ax.set_xticklabels(quarterly["quarter"], rotation=45, ha="right", fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_euro))
    ax.set_title("Quartals-Umsätze  |  Q4 = Saisonhöhepunkt")
    _add_chart_footer(ax)

    plt.tight_layout(pad=3)

    output_path = output_dir / "chart_05_trend_analysis.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart 5 gespeichert: {output_path.name}")
    return output_path


# ─────────────────────────────────────────────
# ALLE CHARTS ERZEUGEN
# ─────────────────────────────────────────────

def create_all_charts(
    monthly_df: pd.DataFrame,
    products_df: pd.DataFrame,
    product_growth_df: pd.DataFrame,
    region_month_df: pd.DataFrame,
    region_channel_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    kpis: dict,
    config: VisualizationConfig,
    output_dir: Path
) -> dict:
    """
    Orchestriert die Erstellung aller Charts.

    Gibt Dictionary mit Pfaden zu allen generierten Chart-Dateien zurück,
    damit der Report-Generator die Charts korrekt referenzieren kann.

    Args:
        ... (alle DataFrames aus Analyse-Phase)
        kpis: KPI-Dictionary
        config: Visualisierungs-Konfiguration
        output_dir: Ausgabe-Pfad

    Returns:
        Dict {chart_name: Path} für alle erzeugten Charts
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style(config)

    chart_paths = {}

    try:
        chart_paths["executive_dashboard"] = create_executive_dashboard(
            monthly_df, kpis, config, output_dir
        )
    except Exception as e:
        logger.error(f"Chart 1 fehlgeschlagen: {e}")

    try:
        chart_paths["product_matrix"] = create_product_matrix(
            product_growth_df, config, output_dir
        )
    except Exception as e:
        logger.error(f"Chart 2 fehlgeschlagen: {e}")

    try:
        chart_paths["regional_heatmap"] = create_regional_heatmap(
            region_month_df, config, output_dir
        )
    except Exception as e:
        logger.error(f"Chart 3 fehlgeschlagen: {e}")

    try:
        chart_paths["rfm_segmentation"] = create_rfm_segmentation(
            rfm_df, config, output_dir
        )
    except Exception as e:
        logger.error(f"Chart 4 fehlgeschlagen: {e}")

    try:
        chart_paths["trend_analysis"] = create_trend_analysis(
            monthly_df, config, output_dir
        )
    except Exception as e:
        logger.error(f"Chart 5 fehlgeschlagen: {e}")

    logger.info(f"✅ {len(chart_paths)}/5 Charts erfolgreich erstellt")
    return chart_paths
