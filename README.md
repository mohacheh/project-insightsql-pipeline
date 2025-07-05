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


🚀 Quickstart
1. Installation
Bash
git clone [https://github.com/dein-nutzername/sql-insights-pipeline.git](https://github.com/dein-nutzername/sql-insights-pipeline.git)
cd sql_insights_pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Ausführung
Die Pipeline bietet einen Demo-Modus, der auch ohne OpenAI API-Key funktioniert:

Bash
# Startet die komplette Pipeline (mit Mock-Insights falls kein Key vorhanden)
python main.py --demo

# Für volle KI-Power: .env Datei erstellen und API-Key eintragen
# python main.py
3. CLI Optionen
--force-recreate-db: Löscht und erstellt die Beispieldatenbank neu.

--skip-charts: Überspringt die Bildgenerierung für schnellere Testläufe.

--log-level DEBUG: Zeigt detaillierte Hintergrundprozesse.

📈 Analyse-Methoden (Business Logic)
Das Tool nutzt anerkannte Frameworks des Controllings:

RFM-Analyse: Segmentierung nach Recency, Frequency, Monetary (Champions vs. At-Risk).

ABC-Analyse: Pareto-Prinzip (80/20 Regel) für die Produktpriorisierung.

BCG-Matrix: Portfolio-Analyse nach Marktwachstum und relativem Marktanteil.

YoY/MoM Growth: Messung der Wachstumsdynamik unter Berücksichtigung von Saisonalität.

📄 Output Beispiele
Nach dem Run finden Sie im /output Ordner:

Executive Dashboard: KPI-Karten & Umsatztrends.

Product Matrix: BCG-Bubble-Charts.

Sales Report (Excel): Vollständig formatierte Daten für Stakeholder.

AI Insights: Ein generierter Text-Report mit konkreten Handlungsempfehlungen.

Beispiel Insight: "Der Champions-Anteil von 19,2% ist stabil, jedoch zeigt die Region Ost einen Rückgang im MoM-Wachstum von 12%. Empfehlung: Re-Engagement Kampagne für Segment 'At Risk'."

🛠 Tech Stack
Core: Python 3.10+

Data: Pandas, NumPy, SQLite

Viz: Matplotlib, Seaborn

AI: OpenAI API (GPT-4o)

Reporting: Openpyxl (Excel), Python-Dotenv

Dieses Projekt dient als Portfolio-Arbeit zur Demonstration von Full-Stack Data-Capabilities.