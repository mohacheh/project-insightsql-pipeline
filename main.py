"""
main.py — SQL-to-Insights Pipeline: Einstiegspunkt & Orchestrierung.

Orchestriert 5 Phasen der Daten-Pipeline:
  1. Setup     → Datenbank + Schema + Beispieldaten
  2. Extract   → SQL-Queries ausführen → DataFrames
  3. Analyze   → Pandas-Analysen + KPI-Berechnung
  4. Visualize → Professionelle Charts erstellen
  5. Report    → Text + Excel Report generieren

Verwendung:
  python main.py                        # Standard-Pipeline
  python main.py --demo                 # Ohne API-Key (Fallback-Insights)
  python main.py --months 12            # Letzten 12 Monate
  python main.py --log-level DEBUG      # Verbose Logging
  python main.py --force-recreate-db    # DB neu aufbauen

Autor: Portfolio-Projekt (Data Engineering + Business Intelligence)
Stack: Python 3.10+ | SQLite | Pandas | Matplotlib/Seaborn | OpenAI
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Projekt-eigene Imports
from config import PipelineConfig, PipelineResult, DEFAULT_CONFIG
from database.setup_db import setup_database
from database.queries import (
    MONTHLY_REVENUE, QUARTERLY_REVENUE, REVENUE_BY_REGION_CHANNEL,
    REVENUE_BY_REGION_MONTH, TOP_PRODUCTS_REVENUE_MARGIN,
    PRODUCT_MONTHLY_REVENUE, CUSTOMER_LIFETIME_VALUE,
    RAW_SALES_FOR_ANALYSIS, KPI_OVERVIEW
)
from analysis.sales_analysis import (
    compute_monthly_growth, compute_revenue_matrix,
    compute_abc_analysis, compute_product_growth_matrix,
    calculate_rfm, calculate_clv_by_segment, detect_churn_risk
)
from analysis.kpi_calculator import calculate_kpis, format_kpi_for_display
from visualization.charts import create_all_charts
from ai_insights.gpt_interpreter import generate_business_insights, extract_trends_for_prompt
from reporting.report_generator import generate_text_report, generate_excel_report
from utils.helpers import setup_logging, load_query, validate_dataframe, print_pipeline_summary

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parst CLI-Argumente für flexible Pipeline-Konfiguration.

    Ermöglicht unterschiedliche Verwendungsszenarien ohne Code-Änderungen:
    - Demo-Modus für Portfolio-Präsentationen
    - Debug-Modus für Entwicklung
    - Filter für spezifische Analysen

    Returns:
        Parsed Namespace mit allen Argumenten
    """
    parser = argparse.ArgumentParser(
        description="SQL-to-Insights Pipeline — Automatisierte Datenanalyse & Reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py                         Standard-Pipeline (alle Phasen)
  python main.py --demo                  Demo-Modus (kein API-Key benötigt)
  python main.py --log-level DEBUG       Detailliertes Logging
  python main.py --force-recreate-db     Datenbank neu aufbauen
  python main.py --skip-charts           Schnell (ohne Visualisierungen)
        """
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo-Modus: Nutzt Fallback-Insights ohne OpenAI API-Key"
    )
    parser.add_argument(
        "--force-recreate-db",
        action="store_true",
        help="Datenbank löschen und neu aufbauen (neue Testdaten)"
    )
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Visualisierungen überspringen (schnellerer Durchlauf)"
    )
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Excel-Export überspringen"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging-Level (default: INFO)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Alternatives Output-Verzeichnis"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# PIPELINE PHASEN
# ─────────────────────────────────────────────

def phase1_setup(config: PipelineConfig, force_recreate: bool = False) -> bool:
    """
    Phase 1: Datenbank-Setup.

    Erstellt SQLite-DB mit Schema und synthetischen Beispieldaten,
    falls noch nicht vorhanden (idempotent).

    Returns:
        True wenn erfolgreich
    """
    logger.info("📦 Phase 1/5: Datenbank Setup...")
    success = setup_database(config.db_path, force_recreate=force_recreate)
    if success:
        logger.info(f"  ✓ Datenbank bereit: {config.db_path}")
    return success


def phase2_extract(config: PipelineConfig) -> dict:
    """
    Phase 2: Daten-Extraktion via SQL.

    Führt alle definierten SQL-Queries aus und gibt bereinigte
    DataFrames zurück. SQL-Fehler werden geloggt, Pipeline läuft weiter.

    Returns:
        Dictionary mit allen rohen DataFrames
    """
    logger.info("🔍 Phase 2/5: Daten extrahieren...")
    data = {}

    queries = {
        "monthly_raw":      MONTHLY_REVENUE,
        "quarterly":        QUARTERLY_REVENUE,
        "region_channel":   REVENUE_BY_REGION_CHANNEL,
        "region_month":     REVENUE_BY_REGION_MONTH,
        "products_raw":     TOP_PRODUCTS_REVENUE_MARGIN,
        "product_monthly":  PRODUCT_MONTHLY_REVENUE,
        "customer_stats":   CUSTOMER_LIFETIME_VALUE,
        "raw_sales":        RAW_SALES_FOR_ANALYSIS,
        "kpi_overview":     KPI_OVERVIEW,
    }

    for name, query in queries.items():
        try:
            df = load_query(config.db_path, query)
            data[name] = df
            logger.info(f"  ✓ {name:<20}: {len(df):>6,} Zeilen")
        except Exception as e:
            logger.error(f"  ✗ {name}: Fehler beim Laden — {e}")
            data[name] = None

    logger.info(f"  → {sum(1 for v in data.values() if v is not None)}/{len(queries)} Queries erfolgreich")
    return data


def phase3_analyze(data: dict, config: PipelineConfig) -> dict:
    """
    Phase 3: Pandas-Analysen + KPI-Berechnung.

    Transformiert rohe DataFrames in analytisch aufbereitete Strukturen:
    - Wachstumsraten (MoM, YoY)
    - ABC-Klassifikation
    - BCG-Matrix
    - RFM-Segmentierung
    - CLV-Berechnung
    - Churn-Risiko-Indikatoren
    - Konsolidierte KPIs

    Returns:
        Dictionary mit allen analysierten DataFrames + KPIs
    """
    logger.info("📊 Phase 3/5: Analyse läuft...")
    results = {}

    # ── Umsatz-Analysen ────────────────────────────────────────
    if validate_dataframe(data.get("monthly_raw"), "monthly_raw", ["month", "revenue"]):
        results["monthly"] = compute_monthly_growth(data["monthly_raw"])
        logger.info("  ✓ Monatliche Wachstumsraten berechnet")

    if validate_dataframe(data.get("region_channel"), "region_channel", ["region", "channel", "revenue"]):
        results["revenue_matrix"] = compute_revenue_matrix(data["region_channel"])
        logger.info("  ✓ Region × Kanal Matrix erstellt")

    # ── Produkt-Analysen ───────────────────────────────────────
    if validate_dataframe(data.get("products_raw"), "products_raw", ["name", "total_revenue", "margin_pct"]):
        results["products_abc"] = compute_abc_analysis(data["products_raw"])
        logger.info(f"  ✓ ABC-Analyse: {len(results['products_abc'])} Produkte klassifiziert")

    if validate_dataframe(data.get("product_monthly"), "product_monthly", ["month", "name", "revenue"]):
        results["product_growth"] = compute_product_growth_matrix(data["product_monthly"])
        logger.info("  ✓ BCG-Matrix berechnet")

    # ── Kunden-Analysen ────────────────────────────────────────
    if validate_dataframe(data.get("raw_sales"), "raw_sales", ["customer_id", "date", "revenue", "id"]):
        results["rfm"] = calculate_rfm(data["raw_sales"])
        logger.info(f"  ✓ RFM-Segmentierung: {len(results['rfm'])} Kunden")

    if validate_dataframe(data.get("customer_stats"), "customer_stats", ["customer_id", "segment"]):
        results["clv_by_segment"] = calculate_clv_by_segment(
            data["customer_stats"], config.analysis
        )
        results["churn_risk"] = detect_churn_risk(data["customer_stats"])
        logger.info("  ✓ CLV und Churn-Risiko berechnet")

    # ── KPI-Konsolidierung ─────────────────────────────────────
    if all(k in results for k in ["monthly", "products_abc", "rfm"]):
        results["kpis"] = calculate_kpis(
            monthly_df=results["monthly"],
            products_df=results["products_abc"],
            region_channel_df=data.get("region_channel", results.get("revenue_matrix", data.get("region_channel"))),
            customer_stats_df=data.get("customer_stats"),
            rfm_df=results["rfm"],
        )
        results["kpi_formatted"] = format_kpi_for_display(results["kpis"])
        logger.info("  ✓ KPIs konsolidiert")

    logger.info(f"  → {len(results)} Analyse-Ergebnisse berechnet")
    return results


def phase4_visualize(
    data: dict,
    results: dict,
    config: PipelineConfig,
    skip: bool = False
) -> dict:
    """
    Phase 4: Charts und Dashboard erstellen.

    Gibt Pfade aller generierten Chart-Dateien zurück.
    Bei skip=True wird Phase übersprungen (für schnelle Tests).

    Returns:
        Dictionary {chart_name: Path}
    """
    if skip:
        logger.info("🎨 Phase 4/5: Charts übersprungen (--skip-charts)")
        return {}

    logger.info("🎨 Phase 4/5: Charts werden erstellt...")

    chart_paths = create_all_charts(
        monthly_df=results.get("monthly"),
        products_df=results.get("products_abc"),
        product_growth_df=results.get("product_growth"),
        region_month_df=data.get("region_month"),
        region_channel_df=data.get("region_channel"),
        rfm_df=results.get("rfm"),
        kpis=results.get("kpis", {}),
        config=config.visualization,
        output_dir=config.output_dir
    )

    logger.info(f"  → {len(chart_paths)} Charts erstellt")
    return chart_paths


def phase5_report(
    data: dict,
    results: dict,
    chart_paths: dict,
    config: PipelineConfig,
    skip_excel: bool = False
) -> dict:
    """
    Phase 5: AI-Insights generieren + Reports erstellen.

    Reihenfolge:
    1. Trend-Signale für AI extrahieren
    2. GPT-Anfrage (oder Fallback)
    3. Text-Report generieren
    4. Excel-Report generieren

    Returns:
        Dictionary {report_type: Path}
    """
    logger.info("📝 Phase 5/5: Report wird generiert...")
    output_files = {}

    # ── AI-Insights generieren ─────────────────────────────────
    logger.info("  🤖 Generiere Business Insights...")
    try:
        trends = extract_trends_for_prompt(results.get("monthly"))
        insights = generate_business_insights(
            kpi_summary=results.get("kpis", {}),
            top_products=results.get("products_abc"),
            regional_performance=data.get("region_channel"),
            trends=trends,
            config=config.ai
        )
        logger.info("  ✓ Business Insights generiert")
    except Exception as e:
        logger.warning(f"  ⚠ Insights-Generierung fehlgeschlagen: {e} — Fallback aktiv")
        insights = "  [Insights konnten nicht generiert werden — Pipeline-Log prüfen]"

    # ── Text-Report ────────────────────────────────────────────
    try:
        text_path = generate_text_report(
            kpis=results.get("kpis", {}),
            kpi_formatted=results.get("kpi_formatted", {}),
            insights=insights,
            monthly_df=results.get("monthly"),
            products_df=results.get("products_abc"),
            rfm_df=results.get("rfm"),
            config=config,
            output_dir=config.output_dir
        )
        output_files["text_report"] = text_path
        logger.info(f"  ✓ Text-Report: {text_path.name}")
    except Exception as e:
        logger.error(f"  ✗ Text-Report fehlgeschlagen: {e}")

    # ── Excel-Report ───────────────────────────────────────────
    if not skip_excel:
        try:
            excel_path = generate_excel_report(
                kpis=results.get("kpis", {}),
                monthly_df=results.get("monthly"),
                products_df=results.get("products_abc"),
                region_channel_df=data.get("region_channel"),
                rfm_df=results.get("rfm"),
                customer_stats_df=data.get("customer_stats"),
                config=config,
                output_dir=config.output_dir
            )
            if excel_path:
                output_files["excel_report"] = excel_path
                logger.info(f"  ✓ Excel-Report: {excel_path.name}")
        except Exception as e:
            logger.error(f"  ✗ Excel-Report fehlgeschlagen: {e}")

    output_files.update(chart_paths)

    logger.info(f"  → {len(output_files)} Output-Dateien generiert")
    return output_files


# ─────────────────────────────────────────────
# PIPELINE ORCHESTRIERUNG
# ─────────────────────────────────────────────

def run_pipeline(config: PipelineConfig, args: argparse.Namespace) -> PipelineResult:
    """
    Orchestriert die komplette SQL-to-Insights Pipeline.

    5 Phasen: Setup → Extract → Analyze → Visualize → Report

    Design-Prinzipien:
    - Fehlertoleranz: Eine fehlgeschlagene Phase stoppt nicht automatisch
    - Transparenz: Jede Phase loggt Status + Key-Metriken
    - Rückgabe: Strukturiertes PipelineResult statt Exceptions

    Args:
        config: PipelineConfig mit allen Einstellungen
        args: Parsed CLI-Argumente

    Returns:
        PipelineResult mit Status, Dateipfaden und KPI-Summary
    """
    start_time = datetime.now()
    result = PipelineResult(success=False, start_time=start_time)

    logger.info("🚀 SQL-to-Insights Pipeline gestartet")
    logger.info(f"   Zeitraum: {config.date_range.start} – {config.date_range.end}")
    logger.info(f"   Output:   {config.output_dir}")
    logger.info(f"   Demo:     {'JA (Fallback-Insights)' if config.demo_mode else 'NEIN (OpenAI aktiv)'}")

    try:
        # Phase 1: Setup
        if not phase1_setup(config, force_recreate=args.force_recreate_db):
            result.error_message = "Datenbank-Setup fehlgeschlagen"
            return result
        result.phases_completed.append("Setup")

        # Phase 2: Extract
        data = phase2_extract(config)
        if not data.get("monthly_raw") is not None:
            result.error_message = "Daten-Extraktion fehlgeschlagen"
            return result
        result.phases_completed.append("Extract")

        # Phase 3: Analyze
        results = phase3_analyze(data, config)
        if not results.get("kpis"):
            logger.warning("KPI-Berechnung unvollständig — Pipeline läuft weiter")
        result.phases_completed.append("Analyze")
        result.kpi_summary = results.get("kpis", {})

        # Phase 4: Visualize
        chart_paths = phase4_visualize(data, results, config, skip=args.skip_charts)
        result.phases_completed.append("Visualize")

        # Phase 5: Report
        output_files = phase5_report(
            data, results, chart_paths, config,
            skip_excel=args.skip_excel
        )
        result.phases_completed.append("Report")
        result.output_files = {k: str(v) for k, v in output_files.items() if v}

        result.success = True
        result.end_time = datetime.now()

        logger.info(f"✅ Pipeline erfolgreich in {result.duration_seconds:.1f}s abgeschlossen")

    except Exception as e:
        logger.exception(f"💥 Unerwarteter Pipeline-Fehler: {e}")
        result.error_message = str(e)
        result.end_time = datetime.now()

    return result


# ─────────────────────────────────────────────
# EINSTIEGSPUNKT
# ─────────────────────────────────────────────

def main() -> int:
    """
    Haupt-Einstiegspunkt mit CLI-Integration.

    Returns:
        Exit-Code: 0 = Erfolg, 1 = Fehler
    """
    args = parse_args()

    # Logging konfigurieren
    log_file = DEFAULT_CONFIG.output_dir / "pipeline.log"
    setup_logging(level=args.log_level, log_file=log_file)

    # Konfiguration anpassen
    config = DEFAULT_CONFIG
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        config.demo_mode = True
        config.ai.fallback_mode = True
        logger.info("Demo-Modus aktiv")

    if args.output_dir:
        config.output_dir = args.output_dir

    # Pipeline ausführen
    result = run_pipeline(config, args)

    # Zusammenfassung ausgeben
    print_pipeline_summary(result)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
