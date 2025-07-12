"""
config.py — Zentrale Konfiguration für die SQL-Insights Pipeline.

Alle Parameter an einem Ort: einfach anpassbar für verschiedene
Umgebungen (Dev / Demo / Prod) ohne Code-Änderungen.
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Projekt-Pfade (pathlib statt os.path — moderner Python-Style)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = BASE_DIR / "database" / "sales.db"


# ─────────────────────────────────────────────
# Pipeline-Konfiguration als Dataclass
# Dataclass statt Dict: Type-Safety + IDE-Autovervollständigung
# ─────────────────────────────────────────────
@dataclass
class DateRangeConfig:
    start: str = "2023-01-01"
    end: str = "2024-12-31"


@dataclass
class AnalysisConfig:
    top_n_products: int = 10
    rfm_segments: int = 5
    moving_average_window: int = 30
    abc_thresholds: tuple = (0.80, 0.95)  # A=80%, B=15%, C=5%
    min_orders_for_clv: int = 2


@dataclass
class VisualizationConfig:
    style: str = "professional"   # professional | minimal | corporate
    dpi: int = 300
    figsize_dashboard: tuple = (20, 14)
    figsize_single: tuple = (12, 8)
    # Konsistente Unternehmensfarben (Corporate Blue Palette)
    colors: dict = field(default_factory=lambda: {
        "primary":   "#1B4F72",
        "secondary": "#2E86C1",
        "accent":    "#3498DB",
        "positive":  "#1E8449",
        "negative":  "#922B21",
        "neutral":   "#7F8C8D",
        "light":     "#EBF5FB",
        "categories": ["#1B4F72", "#2E86C1", "#27AE60", "#E67E22", "#8E44AD"],
    })
    font_family: str = "DejaVu Sans"


@dataclass
class AIConfig:
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 800
    temperature: float = 0.3        # Niedrig = fokussierter, analytischer Output
    fallback_mode: bool = True       # Läuft auch ohne API-Key
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )


@dataclass
class ExportConfig:
    formats: list = field(default_factory=lambda: ["txt", "excel", "png"])
    excel_filename: str = "sales_report.xlsx"
    text_filename: str = "report_{date}.txt"
    chart_prefix: str = "chart"


@dataclass
class PipelineConfig:
    """Master-Konfiguration — wird an alle Pipeline-Phasen weitergereicht."""
    date_range: DateRangeConfig = field(default_factory=DateRangeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    output_dir: Path = OUTPUT_DIR
    db_path: Path = DB_PATH
    report_title: str = "Sales Performance Report"
    company_name: str = "RetailCo GmbH"
    demo_mode: bool = False          # --demo Flag aus CLI


@dataclass
class PipelineResult:
    """Rückgabe-Objekt der Pipeline — strukturierter Status statt lose Variablen."""
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    phases_completed: list = field(default_factory=list)
    output_files: dict = field(default_factory=dict)
    kpi_summary: dict = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# ─────────────────────────────────────────────
# Standard-Konfiguration (wird in main.py genutzt)
# ─────────────────────────────────────────────
DEFAULT_CONFIG = PipelineConfig()
