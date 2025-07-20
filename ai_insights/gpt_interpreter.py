"""
ai_insights/gpt_interpreter.py — KI-gestützte Business-Insight-Generierung.

Integriert OpenAI GPT als "Senior Business Analyst" der die Analyse-
Ergebnisse interpretiert und in Consulting-Report-Sprache überführt.

Design-Entscheidungen:
1. Prompt-Engineering: System-Prompt definiert Persona + Output-Format,
   User-Prompt liefert strukturierte Daten → konsistenter, auswertbarer Output
2. Fallback-Modus: Pipeline läuft ohne API-Key — wichtig für Portfolio-Demos
3. Daten-Kompression: Nur relevante Metriken an API senden (Token-Effizienz)
4. Temperature=0.3: Analytischer, fokussierter Output statt kreative Varianz
"""

import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config import AIConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SYSTEM PROMPT (Persona & Format-Definition)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Du bist ein Senior Business Analyst mit 10+ Jahren Consulting-Erfahrung 
(McKinsey/BCG Niveau). Du analysierst Vertriebsdaten präzise und lieferst 
konkrete, umsetzbare Handlungsempfehlungen.

Dein Output-Format ist IMMER:

## KEY FINDINGS
• [Befund 1: Konkret, datenbasiert, mit Zahlen belegt]
• [Befund 2: Konkret, datenbasiert, mit Zahlen belegt]  
• [Befund 3: Konkret, datenbasiert, mit Zahlen belegt]

## CRITICAL ISSUES
• [Problem mit Business-Impact] — ODER: Keine kritischen Probleme identifiziert.

## RECOMMENDED ACTIONS
1. [Sofortmaßnahme: Was, Warum, erwarteter Impact]
2. [Mittelfristige Maßnahme: 30-90 Tage]
3. [Strategische Empfehlung: 3-6 Monate]

## OUTLOOK
[Ein prägnanter Satz zur Geschäftsentwicklung der nächsten 2 Quartale]

Regeln:
- Keine Floskeln ("Es ist wichtig zu betonen...")
- Zahlen immer mit Kontext ("+12% MoM — deutlich über Marktdurchschnitt")
- Empfehlungen SMART: Specific, Measurable, Achievable, Relevant, Time-bound
- Maximale Präzision, minimale Wortzahl"""


def _build_analysis_prompt(
    kpi_summary: dict,
    top_products: pd.DataFrame,
    regional_performance: pd.DataFrame,
    trends: dict
) -> str:
    """
    Baut strukturierten User-Prompt aus Analyse-Ergebnissen.

    Wichtig: Nur die relevantesten Metriken übergeben.
    Zu viele Daten verschlechtern LLM-Output (Dilution Effect).

    Args:
        kpi_summary: Output von calculate_kpis()
        top_products: Top-10-Produkte DataFrame
        regional_performance: Region × Kanal Matrix (als DataFrame)
        trends: Dict mit Wachstumsraten und Trend-Informationen

    Returns:
        Formatierter Prompt-String für GPT
    """
    rev = kpi_summary.get("revenue", {})
    growth = kpi_summary.get("growth", {})
    customers = kpi_summary.get("customers", {})
    top_perf = kpi_summary.get("top_performers", {})
    period = kpi_summary.get("period", {})

    # Kompakte Produkt-Tabelle (Top 5)
    if len(top_products) > 0:
        product_summary = top_products.head(5)[
            ["name", "category", "total_revenue", "margin_pct", "abc_class"]
        ].to_string(index=False, float_format="%.0f") if "abc_class" in top_products.columns else \
        top_products.head(5)[["name", "category", "total_revenue", "margin_pct"]].to_string(
            index=False, float_format="%.0f"
        )
    else:
        product_summary = "Keine Produktdaten verfügbar"

    prompt = f"""Analysiere die folgende Vertriebsperformance und erstelle einen Executive Summary.

ANALYSE-ZEITRAUM: {period.get('start', 'N/A')} bis {period.get('end', 'N/A')} ({period.get('total_months', 0)} Monate)

═══ KEY PERFORMANCE INDICATORS ═══
Gesamtumsatz:        €{rev.get('total', 0):>12,.0f}
Umsatz (12M):        €{rev.get('last_12_months', 0):>12,.0f}
YoY-Wachstum:        {growth.get('yoy_pct', 'N/A'):>+.1f}% (Jahr-über-Jahr)
Ø MoM-Wachstum:      {growth.get('avg_mom_pct_last6m', 'N/A'):>+.1f}% (Letzte 6M)
Bruttomarge:         {rev.get('gross_margin_pct', 'N/A'):.1f}%
Aktive Kunden:       {customers.get('active_total', 0):>12,}
Champions-Anteil:    {customers.get('champions_pct', 0):.1f}% (Top-RFM-Segment)
At-Risk-Kunden:      {customers.get('at_risk_count', 0):>12,}

═══ TOP-PERFORMER & UNDERPERFORMER ═══
Beste Region:     {top_perf.get('best_region', 'N/A')} (€{top_perf.get('best_region_revenue', 0):,.0f})
Schwächste Region:{top_perf.get('worst_region', 'N/A')} (€{top_perf.get('worst_region_revenue', 0):,.0f})
Bester Kanal:     {top_perf.get('best_channel', 'N/A')}

═══ TOP 5 PRODUKTE ═══
{product_summary}

═══ TREND-SIGNALE ═══
{json.dumps(trends, ensure_ascii=False, indent=2)}

Erstelle deinen Executive Summary im vorgegebenen Format.
Beziehe dich auf konkrete Zahlen aus den Daten oben."""

    return prompt


def generate_business_insights(
    kpi_summary: dict,
    top_products: pd.DataFrame,
    regional_performance: pd.DataFrame,
    trends: dict,
    config: AIConfig
) -> str:
    """
    Sendet Analyse-Ergebnisse an GPT und generiert Business Insights.

    Automatischer Fallback: Bei fehlenden API-Key oder Rate-Limit
    wird ein template-basierter Insight-Text zurückgegeben,
    damit die Pipeline immer vollständig durchläuft.

    Args:
        kpi_summary: KPI-Dictionary aus kpi_calculator
        top_products: Top-Produkte DataFrame
        regional_performance: Region × Kanal Performance
        trends: Trend-Metriken als Dict
        config: AIConfig mit API-Key, Model, Fallback-Flag

    Returns:
        Formatierter Insight-Text im Consulting-Report-Stil
    """
    # Fallback wenn kein API-Key oder Fallback-Modus
    if not config.api_key or config.fallback_mode:
        logger.info("AI-Fallback-Modus aktiv (kein API-Key oder Demo-Modus)")
        return _generate_fallback_insights(kpi_summary, top_products)

    try:
        import openai

        client = openai.OpenAI(api_key=config.api_key)

        user_prompt = _build_analysis_prompt(
            kpi_summary, top_products, regional_performance, trends
        )

        logger.info(f"Sende Anfrage an OpenAI ({config.model})...")

        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        insights = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        logger.info(f"GPT-Response erhalten: {tokens_used} Tokens verwendet")

        return insights

    except ImportError:
        logger.warning("openai Package nicht installiert — Fallback aktiv")
        return _generate_fallback_insights(kpi_summary, top_products)

    except Exception as e:
        logger.warning(f"OpenAI API Fehler: {type(e).__name__}: {e} — Fallback aktiv")
        return _generate_fallback_insights(kpi_summary, top_products)


def _generate_fallback_insights(
    kpi_summary: dict,
    top_products: pd.DataFrame
) -> str:
    """
    Generiert regelbasierte Insights ohne AI — für Demo/Offline-Modus.

    Nutzt direkt die KPI-Daten für konkrete, datenbasierte Aussagen.
    Sieht im Report aus wie AI-generiert, ist aber deterministic.

    Args:
        kpi_summary: KPI-Dictionary
        top_products: Top-Produkte DataFrame

    Returns:
        Formatierter Insight-Text
    """
    rev = kpi_summary.get("revenue", {})
    growth = kpi_summary.get("growth", {})
    customers = kpi_summary.get("customers", {})
    top_perf = kpi_summary.get("top_performers", {})
    products = kpi_summary.get("products", {})

    yoy = growth.get("yoy_pct")
    mom = growth.get("avg_mom_pct_last6m")
    margin = rev.get("gross_margin_pct")
    best_region = top_perf.get("best_region", "N/A")
    worst_region = top_perf.get("worst_region", "N/A")
    best_region_rev = top_perf.get("best_region_revenue", 0)
    worst_region_rev = top_perf.get("worst_region_revenue", 0)
    at_risk = customers.get("at_risk_count", 0)
    champions = customers.get("champions_pct", 0)
    top_product = products.get("top_product_name", "N/A")
    total_rev = rev.get("total", 0)

    # Dynamische Bewertungen
    growth_rating = "stark positiv" if (yoy or 0) > 10 else "moderat positiv" if (yoy or 0) > 0 else "rückläufig"
    margin_rating = "gesund" if (margin or 0) > 30 else "unter Druck"

    yoy_str = f"{yoy:+.1f}%" if yoy is not None else "N/A"
    mom_str = f"{mom:+.1f}%" if mom is not None else "N/A"
    margin_str = f"{margin:.1f}%" if margin is not None else "N/A"
    rev_diff = best_region_rev - worst_region_rev

    insights = f"""## KEY FINDINGS
• **Wachstumsdynamik {growth_rating}**: Mit {yoy_str} YoY-Wachstum übertrifft das Unternehmen \
die typische Einzelhandels-Wachstumsrate von 4-6%. Der Ø MoM-Trend ({mom_str}) deutet auf \
nachhaltigen Momentum-Aufbau hin — besonders im Q4-getriebenen Saisongeschäft.

• **Regionale Performance-Divergenz**: {best_region} führt mit €{best_region_rev:,.0f} deutlich vor \
{worst_region} (€{worst_region_rev:,.0f}) — eine Lücke von €{rev_diff:,.0f} (+{rev_diff/max(worst_region_rev,1)*100:.0f}%). \
Dies signalisiert ungenutzte Wachstumspotenziale in der {worst_region}-Region durch \
gezielte Ressourcenallokation.

• **Kunden-Portfolio-Qualität**: {champions:.1f}% Champions-Anteil im RFM-Modell zeigt solide \
Kundenbindung. {at_risk} Kunden im At-Risk-Segment (keine Aktivität >60 Tage) repräsentieren \
direktes Umsatz-Wiedergewinnungspotenzial durch Reengagement-Kampagnen.

## CRITICAL ISSUES
• **Margen-Monitoring**: Bruttomarge von {margin_str} ist {margin_rating}. \
Produktmix-Verschiebungen hin zu margenstarken Software/Digitalprodukten könnten \
die Gesamtprofitabilität um 2-4 Prozentpunkte verbessern.

## RECOMMENDED ACTIONS
1. **[Sofort — 0-30 Tage]** Reengagement-Kampagne für {at_risk} At-Risk-Kunden: \
Personalisierte Angebote basierend auf Kaufhistorie. Erwarteter Recovery-Rate: 15-25% \
bei €{at_risk * rev.get('avg_order_value', 100) * 0.2:,.0f} potenziellem Umsatz.

2. **[Kurzfristig — 30-90 Tage]** {worst_region}-Region: Vertriebsressourcen aufstocken \
und Best Practices aus {best_region} transferieren. Ziel: Schließung von 25% der \
Performance-Lücke bis Q2.

3. **[Strategisch — 3-6 Monate]** Produktmix-Optimierung: A-Klasse-Produkte (Top-Umsatz, \
Top-Marge) konsequent priorisieren. "{top_product}" als Anker-Produkt für \
Cross-Selling-Kampagnen nutzen.

## OUTLOOK
Bei Fortsetzung des {mom_str} MoM-Trends und erfolgreicher Umsetzung der regionalen \
Expansion ist ein Jahresumsatz von €{total_rev * 1.1:,.0f} (+10%) für das Folgejahr realistisch — \
vorausgesetzt, die saisonale Q4-Stärke kann durch B2B-Kanal-Ausbau verstetigt werden.

---
*[HINWEIS: Insights im Fallback-Modus generiert. Für KI-generierte Analyse: \
OPENAI_API_KEY in .env setzen und ai.fallback_mode=False in config.py]*"""

    return insights


def extract_trends_for_prompt(monthly_df: pd.DataFrame) -> dict:
    """
    Extrahiert kompakte Trend-Signale für den AI-Prompt.

    Konvertiert DataFrame in kompaktes Dict — minimiert Token-Verbrauch
    bei maximaler Informationsdichte für den GPT-Kontext.

    Args:
        monthly_df: Monatliche Umsätze mit Wachstumsraten

    Returns:
        Kompaktes Trend-Dictionary
    """
    df = monthly_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month")

    recent = df.tail(3)
    mom_values = df["mom_growth_pct"].dropna()

    return {
        "letzter_monat_umsatz": round(float(df["revenue"].iloc[-1]), 2),
        "3_monats_wachstum_avg": round(float(recent["mom_growth_pct"].mean()), 2) if len(recent) > 0 else None,
        "mom_trend": "beschleunigend" if len(mom_values) >= 3 and mom_values.iloc[-1] > mom_values.iloc[-3] else "verlangsamt",
        "beste_12_monate": df.nlargest(1, "revenue")["month"].dt.strftime("%Y-%m").iloc[0],
        "schwächste_12_monate": df.nsmallest(1, "revenue")["month"].dt.strftime("%Y-%m").iloc[0],
        "volatilität_pct": round(float(df["revenue"].std() / df["revenue"].mean() * 100), 1),
        "q4_boost_sichtbar": bool(df[df["month"].dt.month.isin([10, 11, 12])]["revenue"].mean() >
                                  df["revenue"].mean() * 1.2),
    }
