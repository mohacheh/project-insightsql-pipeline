# 📊 SQL-to-Insights Pipeline

> **Vollautomatisierte Daten-Pipeline**: Von Rohdaten in SQL über Pandas-Analysen und professionelle Visualisierungen bis hin zu KI-generierten Executive Reports in unter 10 Sekunden.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI GPT-4](https://img.shields.io/badge/AI-GPT--4-green.svg)](https://openai.com/)

Dieses Portfolio-Projekt demonstriert die Symbiose aus **Modern Data Engineering**, **Business Intelligence** und **KI-Integration**. Es löst ein reales Business-Problem: Die Transformation von brachliegenden Transaktionsdaten in strategische Entscheidungsgrundlagen.

---

## 🎯 Value Proposition (für Recruiter & Hiring Manager)

| Kompetenz | Implementierung im Projekt |
| :--- | :--- |
| **Daten-Pipeline-Design** | Modulare 5-Phasen-Architektur (Extraction → Analysis → Visualization → AI → Reporting). |
| **Advanced SQL** | 9 optimierte Queries inkl. Self-Joins, Window Functions und komplexen Aggregationen. |
| **Data Science mit Pandas** | Implementierung von RFM-Segmentierung, ABC-Klassifikation und Moving Averages. |
| **BI & Visualisierung** | 5 Corporate-Style Charts mit Seaborn/Matplotlib (keine Standard-Plots). |
| **KI-Integration** | LLM-Orchestrierung mit OpenAI GPT als "Senior Business Analyst" via Prompt Engineering. |
| **Software Engineering** | Clean Code: Type Hints, Dataclasses, Logging, robuste Fehlerbehandlung & PEP-8. |
| **Business Understanding** | Anwendung von Controlling-Standardmodellen (BCG-Matrix, CLV, Pareto-Prinzip). |

---

## 🏗 Architektur & Struktur

[Image of a data pipeline flowchart showing stages: SQL Database -> Pandas Transformation -> Plotly/Seaborn Charts -> OpenAI API -> Final PDF/Excel Report]

### Projekt-Layout
```text
sql_insights_pipeline/
├── main.py                # Pipeline-Orchestrierung (Entry Point)
├── config.py              # Central Configuration (Dataclasses)
├── database/
│   ├── setup_db.py        # DB-Initialisierung & Synthetische Daten (500+ Kunden)
│   └── queries.py         # Business Logic in SQL (Konstanten)
├── analysis/
│   ├── sales_analysis.py  # Kern-Logik: RFM, ABC, BCG, YoY/MoM
│   └── kpi_calculator.py  # Aggregation von Business-Kennzahlen
├── visualization/
│   └── charts.py          # Export von High-Res Corporate Charts (300 DPI)
├── ai_insights/
│   └── gpt_interpreter.py # OpenAI API Integration & Prompt Logic
├── reporting/
│   └── report_generator.py # Multi-Sheet Excel & Text Report Generation
└── utils/
    └── helpers.py         # Logging, Validation & DB-Utilities

---
