"""
utils/helpers.py — Allgemeine Hilfsfunktionen für die Pipeline.

Sammlung kleiner, wiederverwendbarer Funktionen die in mehreren
Pipeline-Phasen benötigt werden: Logging, DB-Verbindung, Validierung.
"""

import logging
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Konfiguriert Logging für die gesamte Pipeline.

    Zweistufiges Logging: Console (INFO) + optionale Datei (DEBUG).
    Format enthält Timestamp + Logger-Name für bessere Debugging-Erfahrung.

    Args:
        level: Log-Level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional: Pfad zur Log-Datei
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Formatter: Zeitstempel | Level | Modul | Nachricht
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Drittanbieter-Logger auf WARNING setzen (weniger Noise)
    for noisy_lib in ["matplotlib", "PIL", "urllib3", "openai", "httpx"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


def load_query(db_path: Path, query: str, params: tuple = ()) -> pd.DataFrame:
    """
    Führt SQL-Query aus und gibt DataFrame zurück.

    Wrapper um sqlite3 + pd.read_sql_query für konsistente
    Fehlerbehandlung und Connection-Management.

    Args:
        db_path: Pfad zur SQLite-Datenbankdatei
        query: SQL-Query-String
        params: Query-Parameter (für parametrisierte Queries)

    Returns:
        DataFrame mit Query-Ergebnissen

    Raises:
        FileNotFoundError: Wenn Datenbankdatei nicht existiert
        sqlite3.Error: Bei SQL-Fehlern
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Datenbank nicht gefunden: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    required_columns: list[str],
    min_rows: int = 1
) -> bool:
    """
    Validiert DataFrame-Struktur und Mindestgröße.

    Frühe Fehlerkennung verhindert kryptische Fehler downstream.
    Best Practice: Validation an Daten-Grenzen (SQL → Python).

    Args:
        df: Zu validierender DataFrame
        name: Name für Fehlermeldungen
        required_columns: Liste benötigter Spalten
        min_rows: Minimale Zeilenanzahl

    Returns:
        True wenn valide, False bei Problemen (mit Logging)
    """
    logger = logging.getLogger(__name__)

    if df is None or len(df) == 0:
        logger.error(f"Validation fehlgeschlagen [{name}]: DataFrame ist leer")
        return False

    if len(df) < min_rows:
        logger.warning(f"Validation [{name}]: Nur {len(df)} Zeilen (Minimum: {min_rows})")
        return False

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.error(f"Validation fehlgeschlagen [{name}]: Spalten fehlen: {missing}")
        return False

    return True


def ensure_output_dir(output_dir: Path) -> Path:
    """
    Stellt sicher dass Output-Verzeichnis existiert.

    Args:
        output_dir: Zu erstellendes Verzeichnis

    Returns:
        Absoluter Pfad zum Output-Verzeichnis
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def print_pipeline_summary(result) -> None:
    """
    Gibt abschließende Pipeline-Zusammenfassung auf Console aus.

    Args:
        result: PipelineResult Objekt
    """
    print("\n" + "=" * 60)
    print("  PIPELINE ABGESCHLOSSEN")
    print("=" * 60)

    if result.success:
        print(f"  ✅ Status: ERFOLGREICH")
        print(f"  ⏱  Laufzeit: {result.duration_seconds:.1f} Sekunden")
        print(f"  📊 Phasen: {' → '.join(result.phases_completed)}")
        print()
        print("  📁 OUTPUT-DATEIEN:")
        for file_type, path in result.output_files.items():
            if path:
                size_kb = Path(path).stat().st_size / 1024 if Path(path).exists() else 0
                print(f"    {file_type:<20}: {Path(path).name} ({size_kb:.1f} KB)")

        if result.kpi_summary:
            rev = result.kpi_summary.get("revenue", {})
            growth = result.kpi_summary.get("growth", {})
            print()
            print("  💰 KEY METRICS:")
            print(f"    Gesamtumsatz: €{rev.get('total', 0):>12,.2f}")
            yoy = growth.get("yoy_pct")
            if yoy is not None:
                print(f"    YoY-Wachstum: {yoy:>+11.1f}%")
    else:
        print(f"  ❌ Status: FEHLGESCHLAGEN")
        if result.error_message:
            print(f"  💥 Fehler: {result.error_message}")

    print("=" * 60 + "\n")
