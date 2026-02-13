# 📊 SQL-to-Insights Pipeline

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI%20GPT-green.svg)](https://openai.com/)

**Vollautomatisierte Daten-Pipeline**: Von der SQL-Abfrage über komplexe Pandas-Analysen bis hin zu KI-generierten Executive Reports in unter 10 Sekunden.

Dieses Projekt demonstriert die Symbiose aus **Data Engineering**, **Business Intelligence (BI)** und **Künstlicher Intelligenz**. Es transformiert rohe Transaktionsdaten in strategische Entscheidungsgrundlagen.

---

## 🎯 Highlights für Recruiter & Hiring Manager

| Kompetenz | Implementierung im Projekt |
|:---|:---|
| **Data Pipeline Design** | 5-Phasen-Architektur (Setup → Extract → Analyze → Visualize → Report) |
| **Advanced SQL** | 9 komplexe Queries mit Window Functions, Self-Joins und Aggregationen |
| **Business Analytics** | Implementierung von **RFM-Segmentierung**, **ABC-Analyse** und **BCG-Matrix** |
| **AI Integration** | GPT-4o als "Senior Business Analyst" via Prompt Engineering |
| **Software Engineering** | Clean Code (PEP-8), Type Hints, Dataclasses, Logging & Fehlerbehandlung |
| **Professional BI** | Corporate-Design Visualisierungen (Matplotlib/Seaborn) & Multi-Sheet Excel Export |

---

## 🏗 Architektur & Workflow



Die Pipeline ist modular aufgebaut, um Wartbarkeit und Testbarkeit zu gewährleisten:

1.  **Phase 1 (Setup):** Initialisierung einer SQLite DB mit synthetischen Daten (500+ Kunden).
2.  **Phase 2 (Extract):** Ausführung der SQL-Logik (Queries in `database/queries.py`).
3.  **Phase 3 (Analyze):** Statistische Auswertung & KPI-Berechnung mit Pandas/NumPy.
4.  **Phase 4 (Visualize):** Generierung von 5 High-Res Charts (300 DPI).
5.  **Phase 5 (Report):** KI-Interpretation der Daten & finale Dokumentenerstellung.

### Projektstruktur
```text
sql_insights_pipeline/
├── main.py                # Pipeline-Orchestrierung
├── database/              # SQL-Queries & DB-Setup
├── analysis/              # RFM, ABC, BCG & KPI-Logik
├── visualization/         # Corporate Charts (PNG)
├── ai_insights/           # OpenAI Integration
├── reporting/             # Excel- & Text-Generierung
└── output/                # Zielordner für Reports & Charts

```text

# 📊 SQL-to-Insights Pipeline

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI%20GPT--4o-green.svg)](https://openai.com/)

**Vollautomatisierte Daten-Pipeline**: Von der SQL-Abfrage über komplexe Pandas-Analysen bis hin zu KI-generierten Executive Reports in unter 10 Sekunden.

Dieses Projekt demonstriert die Symbiose aus **Data Engineering**, **Business Intelligence (BI)** und **Künstlicher Intelligenz**. Es transformiert rohe Transaktionsdaten in strategische Entscheidungsgrundlagen.

---

## 🏗 Architektur & Workflow

[Image of a data pipeline flowchart showing stages from SQL Extraction to Pandas Analysis to AI Insights Generation]

Die Pipeline folgt einer modularen 5-Phasen-Architektur:

1.  **Setup:** Initialisierung einer SQLite-DB mit synthetischen Daten (500+ Kunden, 2 Jahre Historie).
2.  **Extract:** Ausführung von 9 optimierten SQL-Queries (Joins, Window Functions).
3.  **Analyze:** Statistische Auswertung (RFM, ABC, BCG) mittels Pandas & NumPy.
4.  **Visualize:** Erstellung von 5 High-Res Charts im Corporate-Design (Matplotlib/Seaborn).
5.  **AI Insights:** GPT-4o fungiert als "Senior Business Analyst" und interpretiert die Ergebnisse.

---

## 🚀 Quickstart

### 1. Installation
```bash
# Repository klonen
git clone [https://github.com/dein-nutzername/sql-insights-pipeline.git](https://github.com/dein-nutzername/sql-insights-pipeline.git)
cd sql_insights_pipeline

# Virtual Environment erstellen & aktivieren
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

Viz: Matplotlib, Seaborn

AI: OpenAI API (GPT-4o)

Reporting: Openpyxl (Excel), Python-Dotenv
