"""
analysis/sales_analysis.py — Tiefgehende Pandas-Analysen auf SQL-Rohdaten.

Transformation: Rohdaten → Business-relevante Kennzahlen & Strukturen.

Architektur-Entscheidung: SQL liefert normalisierte Rohdaten,
Python/Pandas übernimmt die analytische Schwerarbeit (RFM, Moving Average,
ABC-Klassifikation, Wachstumsraten). Das ist der typische Stack in
modernen Data-Engineering-Umgebungen (Snowflake + dbt + Python).
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import AnalysisConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _pct_change_safe(new: float, old: float) -> Optional[float]:
    """Prozentuale Veränderung, sicher gegen Division durch 0."""
    if old == 0 or old is None:
        return None
    return round((new - old) / abs(old) * 100, 2)


# ─────────────────────────────────────────────
# A) UMSATZ-ANALYSEN
# ─────────────────────────────────────────────

def compute_monthly_growth(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet MoM- und YoY-Wachstumsraten aus monatlichem Umsatz-DataFrame.

    MoM (Month-over-Month): Kurzfristiger Trend-Indikator.
    YoY (Year-over-Year): Eliminiert Saisonalität → echter Wachstumsindikator.
    Beide Metriken werden in jedem Earnings-Report verwendet.

    Args:
        monthly_df: DataFrame mit Spalten [month, year, revenue, orders, ...]

    Returns:
        Erweiterter DataFrame mit mom_growth_pct, yoy_growth_pct,
        revenue_ma30 (30-Tage Moving Average), cumulative_revenue
    """
    df = monthly_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    # MoM-Wachstum (lag=1)
    df["revenue_lag1"] = df["revenue"].shift(1)
    df["mom_growth_pct"] = (
        (df["revenue"] - df["revenue_lag1"]) / df["revenue_lag1"] * 100
    ).round(2)

    # YoY-Wachstum: gleicher Monat Vorjahr (lag=12)
    df["revenue_lag12"] = df["revenue"].shift(12)
    df["yoy_growth_pct"] = (
        (df["revenue"] - df["revenue_lag12"]) / df["revenue_lag12"] * 100
    ).round(2)

    # Rolling Average (3 Monate) für Trend-Glättung
    # Glättet kurzfristige Ausreißer (z.B. Black Friday) heraus
    df["revenue_ma3"] = df["revenue"].rolling(window=3, min_periods=1).mean().round(2)

    # Rolling Average (6 Monate) für Long-Term Trend
    df["revenue_ma6"] = df["revenue"].rolling(window=6, min_periods=1).mean().round(2)

    # Kumulativer Umsatz für Waterfall-Charts
    df["cumulative_revenue"] = df["revenue"].cumsum().round(2)

    # Quartal als Kategorie
    df["quarter"] = df["month"].dt.to_period("Q").astype(str)

    # Aufräumen: Hilfs-Spalten entfernen
    df = df.drop(columns=["revenue_lag1", "revenue_lag12"])

    logger.info(
        f"Wachstumsraten berechnet: "
        f"Ø MoM={df['mom_growth_pct'].mean():.1f}%, "
        f"letztes YoY={df['yoy_growth_pct'].dropna().iloc[-1]:.1f}%"
    )
    return df


def compute_revenue_matrix(region_channel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt Pivot-Matrix: Region × Kanal → Umsatz.

    Diese Matrix ist in Consulting-Reports allgegenwärtig:
    zeigt sofort welche Region-Kanal-Kombination unter-/überperformt.

    Args:
        region_channel_df: Ergebnis der REVENUE_BY_REGION_CHANNEL Query

    Returns:
        Pivot DataFrame (Regionen als Index, Kanäle als Spalten)
    """
    pivot = region_channel_df.pivot_table(
        index="region",
        columns="channel",
        values="revenue",
        aggfunc="sum",
        fill_value=0
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot.loc["Total"] = pivot.sum()

    return pivot.round(2)


# ─────────────────────────────────────────────
# B) PRODUKT-ANALYSEN
# ─────────────────────────────────────────────

def compute_abc_analysis(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    ABC-Analyse nach Pareto-Prinzip für Produkt-Priorisierung.

    Standard-Methode im Controlling/SCM:
    - A-Produkte: 80% des Umsatzes → höchste Prio, enge Überwachung
    - B-Produkte: nächste 15% → reguläre Prio
    - C-Produkte: letzte 5% → Rationalisierungskandidaten

    Ermöglicht fokussiertes Ressourcen-Management statt "Gießkanne".

    Args:
        products_df: Ergebnis der TOP_PRODUCTS_REVENUE_MARGIN Query

    Returns:
        DataFrame mit abc_class Spalte + kumulative Umsatz-Anteile

    Example:
        >>> abc = compute_abc_analysis(products_raw)
        >>> abc['abc_class'].value_counts()
        A    4
        B    5
        C    11
        dtype: int64
    """
    df = products_df.copy().sort_values("total_revenue", ascending=False)

    total_rev = df["total_revenue"].sum()
    df["revenue_share_pct"] = (df["total_revenue"] / total_rev * 100).round(2)
    df["cumulative_revenue_pct"] = (df["total_revenue"].cumsum() / total_rev).round(4)

    # ABC-Klassifikation mit pd.cut (sauberer als if/elif Kette)
    df["abc_class"] = pd.cut(
        df["cumulative_revenue_pct"],
        bins=[0, 0.80, 0.95, 1.0],
        labels=["A", "B", "C"],
        include_lowest=True
    )

    logger.info(
        "ABC-Analyse: "
        + ", ".join([
            f"{cls}={count} ({df[df.abc_class == cls].total_revenue.sum() / total_rev * 100:.0f}%)"
            for cls, count in df["abc_class"].value_counts().sort_index().items()
        ])
    )
    return df


def compute_product_growth_matrix(product_monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Wachstumsrate je Produkt für BCG-Matrix.

    BCG (Boston Consulting Group) Matrix: Klassisches Strategy-Framework.
    - Stars:        Hoher Umsatz + Hohes Wachstum → investieren
    - Cash Cows:    Hoher Umsatz + Niedriges Wachstum → melken
    - Question Marks: Niedriger Umsatz + Hohes Wachstum → entscheiden
    - Dogs:         Niedriger Umsatz + Niedriges Wachstum → delistieren

    Args:
        product_monthly_df: Monatliche Umsätze je Produkt

    Returns:
        DataFrame mit revenue, growth_rate, bcg_quadrant je Produkt
    """
    df = product_monthly_df.copy()
    df["month"] = pd.to_datetime(df["month"])

    # H1 vs H2 2024 Wachstum als Proxy (H = Halbjahr)
    h1_2024 = df[df["month"].between("2024-01-01", "2024-06-30")]
    h2_2024 = df[df["month"].between("2024-07-01", "2024-12-31")]

    h1_rev = h1_2024.groupby("name")["revenue"].sum()
    h2_rev = h2_2024.groupby("name")["revenue"].sum()

    growth_df = pd.DataFrame({
        "h1_revenue": h1_rev,
        "h2_revenue": h2_rev,
    }).fillna(0)

    growth_df["growth_h1_to_h2"] = (
        (growth_df["h2_revenue"] - growth_df["h1_revenue"]) /
        growth_df["h1_revenue"].replace(0, np.nan) * 100
    ).round(2)

    # Gesamtumsatz 2024
    rev_2024 = df[df["month"].dt.year == 2024].groupby("name")["revenue"].sum()
    growth_df["total_revenue_2024"] = rev_2024

    # Kategorie re-joinen
    cat_map = df.drop_duplicates("name").set_index("name")["category"]
    growth_df["category"] = growth_df.index.map(cat_map)

    # BCG-Quadrant basierend auf Median-Split
    rev_median = growth_df["total_revenue_2024"].median()
    growth_median = growth_df["growth_h1_to_h2"].median()

    def _bcg_quadrant(row):
        high_rev = row["total_revenue_2024"] >= rev_median
        high_growth = row["growth_h1_to_h2"] >= growth_median
        if high_rev and high_growth:   return "Star"
        if high_rev and not high_growth: return "Cash Cow"
        if not high_rev and high_growth: return "Question Mark"
        return "Dog"

    growth_df["bcg_quadrant"] = growth_df.apply(_bcg_quadrant, axis=1)

    return growth_df.dropna(subset=["total_revenue_2024"]).reset_index()


# ─────────────────────────────────────────────
# C) KUNDEN-ANALYSEN
# ─────────────────────────────────────────────

def calculate_rfm(
    df: pd.DataFrame,
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Berechnet RFM-Score (Recency, Frequency, Monetary) für Kundensegmentierung.

    RFM ist der Standard für verhaltensbasierte Kundensegmentierung.
    Jede Dimension wird in Quintile eingeteilt (1-5), höher = besser.
    Gesamtscore 13-15 = Champions, 1-5 = At Risk / Lost.

    Args:
        df: Sales DataFrame mit customer_id, date, revenue Spalten
        reference_date: Referenzdatum für Recency-Berechnung.
                        Default: max(date) im DataFrame

    Returns:
        DataFrame mit rfm_score, rfm_segment, r_score, f_score, m_score

    Example:
        >>> rfm = calculate_rfm(sales_df, datetime(2024, 12, 31))
        >>> print(rfm['rfm_segment'].value_counts())
        Champions         87
        Loyal             124
        At Risk           56
        ...
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if reference_date is None:
        reference_date = df["date"].max()

    # ── RFM-Rohdaten aggregieren ──────────────────────────────
    rfm_raw = df.groupby("customer_id").agg(
        recency=("date", lambda x: (reference_date - x.max()).days),
        frequency=("id", "count"),
        monetary=("revenue", "sum")
    ).reset_index()

    # ── Quintil-Scoring (1-5) ─────────────────────────────────
    # Recency: NIEDRIGERER Wert = besser (Kauf war vor kurzem)
    # → umgekehrte Quintile: Rang 5 = kürzlichster Kauf
    rfm_raw["r_score"] = pd.qcut(
        rfm_raw["recency"],
        q=5,
        labels=[5, 4, 3, 2, 1],  # niedrig = gut → Label umkehren
        duplicates="drop"
    ).astype(int)

    rfm_raw["f_score"] = pd.qcut(
        rfm_raw["frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm_raw["m_score"] = pd.qcut(
        rfm_raw["monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm_raw["rfm_score"] = rfm_raw["r_score"] + rfm_raw["f_score"] + rfm_raw["m_score"]

    # ── Segment-Labels (strategisch benannt für Report) ────────
    def _rfm_segment(row):
        score = row["rfm_score"]
        r = row["r_score"]
        f = row["f_score"]

        if score >= 13:
            return "Champions"
        elif score >= 10 and r >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Recent Customers"
        elif score >= 9:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r == 1 and f >= 3:
            return "Lost Champions"
        else:
            return "Needs Attention"

    rfm_raw["rfm_segment"] = rfm_raw.apply(_rfm_segment, axis=1)

    segment_counts = rfm_raw["rfm_segment"].value_counts()
    logger.info(f"RFM-Segmentierung abgeschlossen:\n{segment_counts.to_string()}")

    return rfm_raw


def calculate_clv_by_segment(
    customer_stats_df: pd.DataFrame,
    config: AnalysisConfig
) -> pd.DataFrame:
    """
    Berechnet Customer Lifetime Value (CLV) je Kundensegment.

    Vereinfachte CLV-Formel: CLV = AOV × Purchase_Frequency × Avg_Lifespan_Years
    (Komplexere Modelle würden Discount Rate und Churn-Wahrscheinlichkeit einbeziehen)

    Args:
        customer_stats_df: Ergebnis der CUSTOMER_LIFETIME_VALUE Query
        config: AnalysisConfig mit min_orders_for_clv Threshold

    Returns:
        DataFrame mit clv_estimate, clv_annualized je Segment
    """
    df = customer_stats_df.copy()
    df = df[df["total_orders"] >= config.min_orders_for_clv]

    # Jährliche Kauffrequenz aus Kundenlaufzeit berechnen
    df["lifespan_years"] = (df["customer_lifespan_days"] / 365).clip(lower=0.1)
    df["annual_purchase_freq"] = df["total_orders"] / df["lifespan_years"]

    # CLV-Schätzung
    df["clv_estimate"] = (
        df["avg_order_value"] * df["annual_purchase_freq"] * df["lifespan_years"]
    ).round(2)

    # Annualisierter CLV (für Vergleichbarkeit)
    df["clv_annualized"] = (df["clv_estimate"] / df["lifespan_years"]).round(2)

    clv_by_segment = df.groupby("segment").agg(
        avg_clv=("clv_estimate", "mean"),
        median_clv=("clv_estimate", "median"),
        avg_annual_clv=("clv_annualized", "mean"),
        avg_orders=("total_orders", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        customer_count=("customer_id", "count")
    ).round(2).sort_values("avg_clv", ascending=False)

    logger.info(f"CLV nach Segment:\n{clv_by_segment[['avg_clv', 'customer_count']].to_string()}")
    return clv_by_segment


def detect_churn_risk(
    customer_stats_df: pd.DataFrame,
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Identifiziert Kunden mit erhöhtem Churn-Risiko.

    Churn-Indikatoren (regelbasiert, kein ML):
    - Kein Kauf in letzten 90 Tagen: hohes Risiko
    - Kein Kauf in 60 Tagen + überdurchschnittlicher historischer Wert: kritisch
    - Kauffrequenz < 50% des Segment-Durchschnitts: mittel

    Args:
        customer_stats_df: Ergebnis der CUSTOMER_LIFETIME_VALUE Query
        reference_date: Referenzdatum, default: heute

    Returns:
        DataFrame mit churn_risk_level (low/medium/high/critical)
    """
    df = customer_stats_df.copy()

    if reference_date is None:
        reference_date = datetime.now()

    df["last_purchase"] = pd.to_datetime(df["last_purchase"])
    df["days_since_last_purchase"] = (reference_date - df["last_purchase"]).dt.days

    # Segment-Durchschnitt für relative Bewertung
    segment_avg_orders = df.groupby("segment")["total_orders"].mean()
    df["segment_avg_orders"] = df["segment"].map(segment_avg_orders)
    df["relative_frequency"] = df["total_orders"] / df["segment_avg_orders"]

    avg_revenue = df["total_revenue"].mean()

    def _churn_risk(row):
        days = row["days_since_last_purchase"]
        is_high_value = row["total_revenue"] > avg_revenue

        if days > 180:
            return "critical" if is_high_value else "high"
        elif days > 90:
            return "high" if is_high_value else "medium"
        elif days > 60 and row["relative_frequency"] < 0.5:
            return "medium"
        else:
            return "low"

    df["churn_risk"] = df.apply(_churn_risk, axis=1)

    risk_dist = df["churn_risk"].value_counts()
    logger.info(f"Churn-Risiko Verteilung:\n{risk_dist.to_string()}")

    return df[["customer_id", "segment", "total_revenue", "total_orders",
               "days_since_last_purchase", "churn_risk"]].sort_values(
        ["churn_risk", "total_revenue"], ascending=[True, False]
    )
