"""
database/setup_db.py — Datenbank-Setup & realistische Beispieldaten.

Generiert ein synthetisches E-Commerce-Dataset mit eingebetteten Mustern:
- Saisonalität (Q4-Boost, Sommer-Delle)
- Wachstumstrend (~5% MoM)
- Black-Friday-Ausreißer
- Regionale Performance-Unterschiede
- Kunden-Segmentierung mit realistischen Kaufmuster-Unterschieden

Diese Daten-Generierungs-Logik ist selbst ein Portfolio-Statement:
zeigt Verständnis für realistische Datenstrukturen, die BI-Tools
herausfordern.
"""

import sqlite3
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Produkt-Master (20 Produkte, 4 Kategorien)
# Reale Preisstrukturen: Electronics teurer + höhere Marge
# ─────────────────────────────────────────────
PRODUCTS = [
    # (id, name, category, price, cost)
    (1,  "Laptop Pro 15",        "Electronics",  1299.00, 780.00),
    (2,  "Wireless Mouse",       "Electronics",    45.00,  12.00),
    (3,  "USB-C Hub",            "Electronics",    79.00,  22.00),
    (4,  "Mechanical Keyboard",  "Electronics",   149.00,  55.00),
    (5,  "Monitor 27 4K",        "Electronics",   599.00, 310.00),
    (6,  "Office Chair Ergo",    "Furniture",     449.00, 180.00),
    (7,  "Standing Desk",        "Furniture",     699.00, 290.00),
    (8,  "Desk Organizer",       "Furniture",      39.00,  10.00),
    (9,  "Bookshelf Nordic",     "Furniture",     199.00,  75.00),
    (10, "LED Desk Lamp",        "Furniture",      59.00,  18.00),
    (11, "Python Course Pro",    "Software",      199.00,   5.00),
    (12, "Data Analytics Suite", "Software",      499.00,  15.00),
    (13, "Project Mgmt Tool",    "Software",       99.00,   3.00),
    (14, "Security Suite",       "Software",      149.00,   8.00),
    (15, "Design Bundle",        "Software",      299.00,  10.00),
    (16, "Coffee Maker Pro",     "Appliances",    129.00,  45.00),
    (17, "Air Purifier",         "Appliances",    249.00,  90.00),
    (18, "Smart Thermostat",     "Appliances",    199.00,  80.00),
    (19, "Robot Vacuum",         "Appliances",    399.00, 160.00),
    (20, "Noise Cancel Headset", "Electronics",   299.00,  95.00),
]

REGIONS = ["Nord", "Süd", "Ost", "West"]
CHANNELS = ["Online", "Stationär", "B2B"]
CUSTOMER_SEGMENTS = ["Premium", "Standard", "Budget"]

# Regionale Performance-Multiplikatoren (West stärker, Ost schwächer)
REGION_MULTIPLIERS = {"Nord": 1.05, "Süd": 1.15, "Ost": 0.85, "West": 1.20}

# Kanal-Multiplikatoren (B2B = große Orders, geringere Frequenz)
CHANNEL_MULTIPLIERS = {"Online": 1.0, "Stationär": 0.90, "B2B": 1.35}

# Segment-Multiplikatoren (Premium kauft mehr & öfter)
SEGMENT_MULTIPLIERS = {"Premium": 1.40, "Standard": 1.00, "Budget": 0.65}


def _get_seasonality_factor(date: datetime) -> float:
    """
    Berechnet saisonalen Umsatz-Multiplikator basierend auf Datum.

    Eingebettete Muster:
    - Q4 (Okt-Dez): +30-50% (Weihnachtsgeschäft)
    - Black Friday (4. Freitag Nov): +300%
    - Sommer (Jun-Aug): leicht schwächer (-10%)
    - Januar-Delle: -20% (Post-Weihnachts-Effekt)

    Args:
        date: Das zu bewertende Datum

    Returns:
        Multiplikator zwischen 0.7 und 4.0
    """
    month = date.month

    # Saisonale Basis-Faktoren
    seasonal_base = {
        1: 0.78,   # Januar-Delle (Post-Weihnachten, Budgets erschöpft)
        2: 0.85,   # Valentinstag leichter Boost
        3: 0.92,   # Vorfrühling
        4: 0.95,
        5: 1.00,   # Baseline
        6: 0.90,   # Sommerloch beginnt
        7: 0.85,   # Hochsommer - Urlaub
        8: 0.88,
        9: 1.05,   # Herbst-Comeback, Back-to-School
        10: 1.15,  # Q4-Anlauf
        11: 1.35,  # November Vorweihnacht
        12: 1.50,  # Dezember Hochsaison
    }

    factor = seasonal_base.get(month, 1.0)

    # Black Friday: 4. Freitag im November
    if month == 11:
        # Prüfe ob Datum im Black-Friday-Fenster liegt (Woche des 4. Freitags)
        fridays_in_nov = [
            d for d in pd.date_range(f"{date.year}-11-01", f"{date.year}-11-30")
            if d.weekday() == 4  # 4 = Freitag
        ]
        if len(fridays_in_nov) >= 4:
            black_friday = fridays_in_nov[3]  # 4. Freitag
            days_diff = abs((date - black_friday.to_pydatetime()).days)
            if days_diff <= 3:  # Black Friday Wochenende
                factor *= 4.0

    return factor


def _get_growth_factor(date: datetime, start_date: datetime) -> float:
    """
    Berechnet kumulativen Wachstumsfaktor (~5% MoM compound growth).

    Repräsentiert ein realistisches, gesundes Unternehmenswachstum
    das Recruitern zeigt: ich verstehe Business-Kontext, nicht nur Technik.

    Args:
        date: Aktuelles Datum
        start_date: Startdatum der Zeitreihe

    Returns:
        Compound-Growth-Faktor (z.B. 1.05^12 ≈ 1.80 nach einem Jahr)
    """
    months_elapsed = (
        (date.year - start_date.year) * 12 +
        (date.month - start_date.month)
    )
    # 3% MoM Growth (realistischer als 5% über 24 Monate)
    return (1.03) ** max(0, months_elapsed)


def generate_customers(n: int = 500, seed: int = 42) -> list[tuple]:
    """
    Generiert synthetische Kundenstammdaten.

    Segment-Verteilung: Premium 20% | Standard 55% | Budget 25%
    (typische Pareto-Verteilung im Retail)

    Args:
        n: Anzahl Kunden
        seed: Random seed für Reproduzierbarkeit

    Returns:
        Liste von (id, segment, region, acquisition_channel) Tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    segment_weights = [0.20, 0.55, 0.25]  # Premium, Standard, Budget
    customers = []

    for i in range(1, n + 1):
        segment = random.choices(CUSTOMER_SEGMENTS, weights=segment_weights)[0]
        region = random.choice(REGIONS)

        # Akquisitionskanal: B2B-Kunden kommen selten über Stationär
        if segment == "Premium":
            channel = random.choices(CHANNELS, weights=[0.50, 0.20, 0.30])[0]
        elif segment == "Budget":
            channel = random.choices(CHANNELS, weights=[0.60, 0.35, 0.05])[0]
        else:
            channel = random.choices(CHANNELS, weights=[0.50, 0.35, 0.15])[0]

        customers.append((i, segment, region, channel))

    return customers


def generate_sales(
    customers: list[tuple],
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42
) -> list[tuple]:
    """
    Generiert synthetische Verkaufsdaten mit realistischen Mustern.

    Eingebettete Komplexität:
    - Saisonalität + Wachstumstrend (multiplikativ kombiniert)
    - Kunden-Segment-spezifisches Kaufverhalten
    - Regionale Performance-Unterschiede
    - Produkt-Kategorie-Affinität je Segment
    - Noise für realistische Varianz

    Args:
        customers: Liste von Kundenstammdaten
        start_date: Startdatum (YYYY-MM-DD)
        end_date: Enddatum (YYYY-MM-DD)
        seed: Random seed

    Returns:
        Liste von Sales-Records als Tuples

    Example:
        >>> customers = generate_customers(100)
        >>> sales = generate_sales(customers, "2024-01-01", "2024-06-30")
        >>> len(sales) > 0
        True
    """
    random.seed(seed)
    np.random.seed(seed)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days

    products_dict = {p[0]: p for p in PRODUCTS}

    # Kategorie-Präferenzen je Segment
    # Premium mag Software + High-End Electronics
    segment_category_prefs = {
        "Premium":  {"Electronics": 0.35, "Software": 0.30, "Furniture": 0.25, "Appliances": 0.10},
        "Standard": {"Electronics": 0.30, "Software": 0.20, "Furniture": 0.30, "Appliances": 0.20},
        "Budget":   {"Electronics": 0.20, "Software": 0.15, "Furniture": 0.35, "Appliances": 0.30},
    }

    sales = []
    sale_id = 1

    for customer in customers:
        cust_id, segment, region, acq_channel = customer

        # Basis-Kauffrequenz pro Jahr je Segment
        annual_orders_base = {"Premium": 18, "Standard": 8, "Budget": 4}
        n_orders = int(np.random.poisson(annual_orders_base[segment] * 2))  # 2 Jahre
        n_orders = max(1, n_orders)

        seg_mult = SEGMENT_MULTIPLIERS[segment]
        region_mult = REGION_MULTIPLIERS[region]

        for _ in range(n_orders):
            # Zufälliges Datum im Zeitraum
            random_day = random.randint(0, total_days)
            sale_date = start_dt + timedelta(days=random_day)

            # Kanal des Kaufs (meist Akquisitionskanal, aber nicht immer)
            channel_probs = {"Online": 0.5, "Stationär": 0.3, "B2B": 0.2}
            # Kunden bleiben tendenziell ihrem Akquisitionskanal treu
            channel_probs[acq_channel] = min(0.7, channel_probs.get(acq_channel, 0.3) + 0.3)
            total_prob = sum(channel_probs.values())
            channel_probs = {k: v / total_prob for k, v in channel_probs.items()}

            sale_channel = random.choices(
                list(channel_probs.keys()),
                weights=list(channel_probs.values())
            )[0]
            channel_mult = CHANNEL_MULTIPLIERS[sale_channel]

            # Produkt-Auswahl basierend auf Segment-Affinität
            prefs = segment_category_prefs[segment]
            categories = list(prefs.keys())
            cat_weights = list(prefs.values())
            chosen_category = random.choices(categories, weights=cat_weights)[0]

            category_products = [p for p in PRODUCTS if p[2] == chosen_category]
            product = random.choice(category_products)
            product_id, _, _, price, cost = product

            # Menge: B2B kauft mehr Einheiten
            if sale_channel == "B2B":
                quantity = int(np.random.poisson(5)) + 1
            elif segment == "Budget":
                quantity = 1
            else:
                quantity = int(np.random.poisson(1.5)) + 1

            quantity = max(1, quantity)

            # Saisonalität + Wachstum als Preis-/Volumen-Multiplikator
            seasonal_factor = _get_seasonality_factor(sale_date)
            growth_factor = _get_growth_factor(sale_date, start_dt)

            # Revenue mit allen Faktoren + Gaussian Noise
            base_revenue = price * quantity
            noise = np.random.normal(1.0, 0.08)  # ±8% Noise
            revenue = round(
                base_revenue * seg_mult * region_mult * channel_mult
                * seasonal_factor * growth_factor * max(0.5, noise),
                2
            )

            sales.append((
                sale_id,
                sale_date.strftime("%Y-%m-%d"),
                product_id,
                cust_id,
                quantity,
                revenue,
                region,
                sale_channel
            ))
            sale_id += 1

    logger.info(f"Generiert: {len(sales):,} Sales-Records für {len(customers)} Kunden")
    return sales


def setup_database(db_path: Path, force_recreate: bool = False) -> bool:
    """
    Erstellt SQLite-Datenbank mit Schema und lädt Beispieldaten.

    Idempotent: Wird die DB bereits gefunden, wird sie übersprungen
    (außer force_recreate=True) — wichtig für schnelle Dev-Iterationen.

    Args:
        db_path: Pfad zur SQLite-Datenbankdatei
        force_recreate: DB löschen und neu aufbauen wenn True

    Returns:
        True wenn erfolgreich, False bei Fehler
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not force_recreate:
        logger.info(f"Datenbank bereits vorhanden: {db_path}")
        return True

    logger.info(f"Erstelle Datenbank: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ─── Schema ───────────────────────────────────────────────
        cursor.executescript("""
            DROP TABLE IF EXISTS sales;
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS customers;

            CREATE TABLE products (
                id       INTEGER PRIMARY KEY,
                name     TEXT    NOT NULL,
                category TEXT    NOT NULL,
                price    REAL    NOT NULL,
                cost     REAL    NOT NULL
            );

            CREATE TABLE customers (
                id                  INTEGER PRIMARY KEY,
                segment             TEXT    NOT NULL,
                region              TEXT    NOT NULL,
                acquisition_channel TEXT    NOT NULL
            );

            CREATE TABLE sales (
                id          INTEGER PRIMARY KEY,
                date        TEXT    NOT NULL,
                product_id  INTEGER NOT NULL REFERENCES products(id),
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                quantity    INTEGER NOT NULL,
                revenue     REAL    NOT NULL,
                region      TEXT    NOT NULL,
                channel     TEXT    NOT NULL
            );

            -- Index für häufige Abfragen (date, region, product)
            CREATE INDEX idx_sales_date     ON sales(date);
            CREATE INDEX idx_sales_region   ON sales(region);
            CREATE INDEX idx_sales_product  ON sales(product_id);
            CREATE INDEX idx_sales_customer ON sales(customer_id);
        """)
        logger.info("Schema erstellt")

        # ─── Produkte laden ────────────────────────────────────────
        cursor.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
            PRODUCTS
        )
        logger.info(f"Produkte geladen: {len(PRODUCTS)}")

        # ─── Kunden generieren & laden ────────────────────────────
        customers = generate_customers(n=500)
        cursor.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            customers
        )
        logger.info(f"Kunden geladen: {len(customers)}")

        # ─── Sales generieren & laden ─────────────────────────────
        sales = generate_sales(customers)
        cursor.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            sales
        )
        logger.info(f"Sales geladen: {len(sales):,}")

        conn.commit()
        conn.close()

        logger.info("✅ Datenbank-Setup abgeschlossen")
        return True

    except Exception as e:
        logger.error(f"Fehler beim Datenbank-Setup: {e}")
        if db_path.exists():
            db_path.unlink()
        return False
