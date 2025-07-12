"""
analysis/kpi_calculator.py — Automatische Business-KPI Berechnung.

KPIs (Key Performance Indicators) sind die Sprache zwischen Technik und
Business. Dieser Layer übersetzt Rohdaten in Kennzahlen, die C-Level
und Consultants verstehen und in Entscheidungen überführen können.

Design: Alle KPIs in einem strukturierten Dict → einfach an AI-Layer
weiterzugeben UND direkt im Report zu formatieren.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_kpis(
    monthly_df: pd.DataFrame,
    products_df: pd.DataFrame,
    region_channel_df: pd.DataFrame,
    customer_stats_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    reference_date: Optional[datetime] = None
) -> dict:
    """
    Berechnet alle Business-KPIs aus den verschiedenen Analyse-DataFrames.

    Konsolidiert alle Kennzahlen in einem strukturierten Dictionary,
    das sowohl für Report-Generierung als auch AI-Prompt-Kontext dient.

    Args:
        monthly_df: Monatliche Umsatzdaten mit Wachstumsraten
        products_df: Produkt-Performance inkl. ABC-Klassifikation
        region_channel_df: Region × Kanal Matrix
        customer_stats_df: Kundenstamm mit CLV-Daten
        rfm_df: RFM-Segmentierung
        reference_date: Stichtag, default: letzter Datenmonat

    Returns:
        Strukturiertes KPI-Dictionary mit folgenden Sektionen:
        - revenue: Umsatzkennzahlen
        - growth: Wachstumsraten
        - products: Produktperformance
        - customers: Kundenkennzahlen
        - operations: Operative KPIs
        - top_performers: Best/Worst je Dimension
        - period: Zeitraum-Info
    """
    monthly_df = monthly_df.copy()
    monthly_df["month"] = pd.to_datetime(monthly_df["month"])

    if reference_date is None:
        reference_date = monthly_df["month"].max()

    # Letzte 12 Monate für "Current Period" KPIs
    cutoff_12m = reference_date - pd.DateOffset(months=12)
    recent_12m = monthly_df[monthly_df["month"] > cutoff_12m]
    prev_12m = monthly_df[
        (monthly_df["month"] > reference_date - pd.DateOffset(months=24)) &
        (monthly_df["month"] <= cutoff_12m)
    ]

    # ── Revenue KPIs ──────────────────────────────────────────
    total_revenue = monthly_df["revenue"].sum()
    revenue_12m = recent_12m["revenue"].sum()
    revenue_prev_12m = prev_12m["revenue"].sum()
    total_orders = monthly_df["orders"].sum()

    revenue_yoy = (
        (revenue_12m - revenue_prev_12m) / revenue_prev_12m * 100
        if revenue_prev_12m > 0 else None
    )

    # ── Gross Margin aus Produkt-Daten ────────────────────────
    if "total_revenue" in products_df.columns and "margin_pct" in products_df.columns:
        weighted_margin = (
            (products_df["total_revenue"] * products_df["margin_pct"]).sum() /
            products_df["total_revenue"].sum()
        )
    else:
        weighted_margin = None

    # ── Growth KPIs ───────────────────────────────────────────
    recent_mom_values = monthly_df["mom_growth_pct"].dropna().tail(6)
    avg_mom_growth = recent_mom_values.mean()

    recent_yoy_values = monthly_df["yoy_growth_pct"].dropna().tail(3)
    avg_yoy_growth = recent_yoy_values.mean()

    # ── Produkt KPIs ──────────────────────────────────────────
    a_products = products_df[products_df.get("abc_class", pd.Series()) == "A"] if "abc_class" in products_df.columns else pd.DataFrame()
    top_product = products_df.iloc[0] if len(products_df) > 0 else None
    worst_product = products_df.iloc[-1] if len(products_df) > 0 else None

    # ── Kunden KPIs ───────────────────────────────────────────
    active_customers = customer_stats_df["customer_id"].nunique()
    avg_clv = customer_stats_df["total_revenue"].mean()
    avg_order_value = customer_stats_df["avg_order_value"].mean()

    # Beste RFM-Segmente
    champions_count = len(rfm_df[rfm_df["rfm_segment"] == "Champions"]) if "rfm_segment" in rfm_df.columns else 0
    at_risk_count = len(rfm_df[rfm_df["rfm_segment"] == "At Risk"]) if "rfm_segment" in rfm_df.columns else 0

    # ── Region / Kanal Top-Performer ─────────────────────────
    region_totals = region_channel_df.groupby("region")["revenue"].sum()
    best_region = region_totals.idxmax() if len(region_totals) > 0 else "N/A"
    worst_region = region_totals.idxmin() if len(region_totals) > 0 else "N/A"

    channel_totals = region_channel_df.groupby("channel")["revenue"].sum()
    best_channel = channel_totals.idxmax() if len(channel_totals) > 0 else "N/A"

    # ── Zusammenstellung ──────────────────────────────────────
    kpis = {
        "period": {
            "start": monthly_df["month"].min().strftime("%Y-%m"),
            "end": monthly_df["month"].max().strftime("%Y-%m"),
            "total_months": len(monthly_df),
            "reference_date": reference_date.strftime("%Y-%m-%d"),
        },
        "revenue": {
            "total": round(total_revenue, 2),
            "last_12_months": round(revenue_12m, 2),
            "prev_12_months": round(revenue_prev_12m, 2),
            "total_orders": int(total_orders),
            "gross_margin_pct": round(weighted_margin, 1) if weighted_margin else None,
            "avg_order_value": round(avg_order_value, 2),
        },
        "growth": {
            "yoy_pct": round(revenue_yoy, 1) if revenue_yoy else None,
            "avg_mom_pct_last6m": round(avg_mom_growth, 1) if not np.isnan(avg_mom_growth) else None,
            "avg_yoy_pct_last3m": round(avg_yoy_growth, 1) if not np.isnan(avg_yoy_growth) else None,
        },
        "products": {
            "total_products": len(products_df),
            "a_class_count": len(a_products),
            "top_product_name": top_product["name"] if top_product is not None else "N/A",
            "top_product_revenue": round(float(top_product["total_revenue"]), 2) if top_product is not None else 0,
            "top_product_margin": round(float(top_product["margin_pct"]), 1) if top_product is not None else 0,
            "worst_product_name": worst_product["name"] if worst_product is not None else "N/A",
        },
        "customers": {
            "active_total": int(active_customers),
            "avg_clv": round(avg_clv, 2),
            "champions_count": int(champions_count),
            "at_risk_count": int(at_risk_count),
            "champions_pct": round(champions_count / active_customers * 100, 1) if active_customers > 0 else 0,
        },
        "top_performers": {
            "best_region": best_region,
            "worst_region": worst_region,
            "best_channel": best_channel,
            "best_region_revenue": round(float(region_totals.max()), 2) if len(region_totals) > 0 else 0,
            "worst_region_revenue": round(float(region_totals.min()), 2) if len(region_totals) > 0 else 0,
        }
    }

    # ── Logging-Summary für schnellen Überblick ───────────────
    logger.info("=" * 50)
    logger.info("KPI SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total Revenue:    €{total_revenue:>12,.2f}")
    logger.info(f"Revenue (12M):    €{revenue_12m:>12,.2f}  (YoY: {revenue_yoy:+.1f}%)" if revenue_yoy else f"Revenue (12M):    €{revenue_12m:>12,.2f}")
    logger.info(f"Ø MoM Growth:      {avg_mom_growth:>11.1f}%")
    logger.info(f"Active Customers: {active_customers:>12,}")
    logger.info(f"Ø Order Value:    €{avg_order_value:>12,.2f}")
    logger.info(f"Best Region:       {best_region}")
    logger.info("=" * 50)

    return kpis


def format_kpi_for_display(kpis: dict) -> dict:
    """
    Formatiert KPI-Werte für Text-Report und Excel-Export.

    Trennung von Berechnung und Formatierung (SRP: Single Responsibility Principle).
    Zahlenformat: European style (Punkte als Tausendertrennzeichen, Komma für Dezimal)
    → angepasst für deutschen Business-Kontext.

    Args:
        kpis: Output von calculate_kpis()

    Returns:
        Dict mit formatierten String-Werten für Display
    """
    def fmt_eur(val: float) -> str:
        if val is None:
            return "N/A"
        return f"€{val:>12,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def fmt_pct(val: float, with_sign: bool = False) -> str:
        if val is None:
            return "N/A"
        sign = "+" if val > 0 and with_sign else ""
        return f"{sign}{val:.1f}%"

    def fmt_int(val: int) -> str:
        if val is None:
            return "N/A"
        return f"{val:,}".replace(",", ".")

    rev = kpis["revenue"]
    growth = kpis["growth"]
    customers = kpis["customers"]

    return {
        "total_revenue":      fmt_eur(rev["total"]),
        "revenue_12m":        fmt_eur(rev["last_12_months"]),
        "yoy_growth":         fmt_pct(growth["yoy_pct"], with_sign=True),
        "mom_growth_avg":     fmt_pct(growth["avg_mom_pct_last6m"], with_sign=True),
        "gross_margin":       fmt_pct(rev["gross_margin_pct"]),
        "avg_order_value":    fmt_eur(rev["avg_order_value"]),
        "active_customers":   fmt_int(customers["active_total"]),
        "champions_pct":      fmt_pct(customers["champions_pct"]),
        "best_region":        kpis["top_performers"]["best_region"],
        "worst_region":       kpis["top_performers"]["worst_region"],
        "best_channel":       kpis["top_performers"]["best_channel"],
        "top_product":        kpis["products"]["top_product_name"],
    }
