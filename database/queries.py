"""
database/queries.py — Alle SQL-Abfragen als benannte Konstanten.

Design-Entscheidung: SQL als Strings in einer dedizierten Datei statt
inline im Analyse-Code. Vorteile:
- SQL leicht austauschbar gegen echte DB (Snowflake, BigQuery, etc.)
- Queries testbar & versionierbar
- Klare Trennung: SQL-Schicht vs. Python-Analyse-Schicht
  (entspricht dem Schichtenmodell in professionellen BI-Stacks)
"""

# ─────────────────────────────────────────────────────────────
# REVENUE QUERIES
# ─────────────────────────────────────────────────────────────

MONTHLY_REVENUE = """
    -- Monatliche Umsätze mit YoY-Vergleich via Self-Join
    -- strftime() = SQLite-Äquivalent zu DATE_TRUNC in PostgreSQL
    SELECT
        strftime('%Y-%m', s.date)               AS month,
        strftime('%Y', s.date)                  AS year,
        CAST(strftime('%m', s.date) AS INTEGER)  AS month_num,
        SUM(s.revenue)                          AS revenue,
        COUNT(*)                                AS orders,
        COUNT(DISTINCT s.customer_id)           AS unique_customers,
        AVG(s.revenue)                          AS avg_order_value
    FROM sales s
    GROUP BY 1
    ORDER BY 1
"""

QUARTERLY_REVENUE = """
    SELECT
        strftime('%Y', date)                AS year,
        CASE
            WHEN CAST(strftime('%m', date) AS INTEGER) BETWEEN 1 AND 3  THEN 'Q1'
            WHEN CAST(strftime('%m', date) AS INTEGER) BETWEEN 4 AND 6  THEN 'Q2'
            WHEN CAST(strftime('%m', date) AS INTEGER) BETWEEN 7 AND 9  THEN 'Q3'
            ELSE 'Q4'
        END                                 AS quarter,
        SUM(revenue)                        AS revenue,
        COUNT(*)                            AS orders,
        COUNT(DISTINCT customer_id)         AS unique_customers
    FROM sales
    GROUP BY 1, 2
    ORDER BY 1, 2
"""

REVENUE_BY_REGION_CHANNEL = """
    -- Kreuztabelle: Region × Kanal — zeigt Multi-Dimensional-Analyse
    SELECT
        region,
        channel,
        SUM(revenue)                AS revenue,
        COUNT(*)                    AS orders,
        AVG(revenue)                AS avg_order_value,
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM sales
    GROUP BY region, channel
    ORDER BY revenue DESC
"""

REVENUE_BY_REGION_MONTH = """
    -- Für Heatmap: Monat × Region → Umsatz
    SELECT
        strftime('%Y-%m', date)  AS month,
        region,
        SUM(revenue)             AS revenue
    FROM sales
    GROUP BY 1, 2
    ORDER BY 1, 2
"""

# ─────────────────────────────────────────────────────────────
# PRODUCT QUERIES
# ─────────────────────────────────────────────────────────────

TOP_PRODUCTS_REVENUE_MARGIN = """
    -- Top-Produkte: Umsatz UND Marge — beide Dimensionen wichtig!
    -- Hoher Umsatz + niedrige Marge = Achtung (Commodities)
    -- Niedriger Umsatz + hohe Marge = Wachstumshebel
    SELECT
        p.id,
        p.name,
        p.category,
        p.price,
        p.cost,
        (p.price - p.cost)          AS gross_profit_per_unit,
        ROUND((p.price - p.cost) / p.price * 100, 1)  AS margin_pct,
        SUM(s.revenue)              AS total_revenue,
        SUM(s.quantity)             AS total_units,
        COUNT(*)                    AS total_orders,
        AVG(s.revenue)              AS avg_order_revenue
    FROM products p
    JOIN sales s ON p.id = s.product_id
    GROUP BY p.id
    ORDER BY total_revenue DESC
"""

PRODUCT_MONTHLY_REVENUE = """
    -- Zeitreihe je Produkt → für Kategorie-Shift-Analyse
    SELECT
        strftime('%Y-%m', s.date)  AS month,
        p.category,
        p.name,
        SUM(s.revenue)             AS revenue,
        SUM(s.quantity)            AS units
    FROM sales s
    JOIN products p ON s.product_id = p.id
    GROUP BY 1, 2, 3
    ORDER BY 1, 2, 3
"""

CATEGORY_REVENUE_SHARE = """
    SELECT
        p.category,
        SUM(s.revenue)              AS revenue,
        COUNT(*)                    AS orders,
        ROUND(SUM(s.revenue) * 100.0 / (SELECT SUM(revenue) FROM sales), 2) AS revenue_share_pct
    FROM products p
    JOIN sales s ON p.id = s.product_id
    GROUP BY p.category
    ORDER BY revenue DESC
"""

# ─────────────────────────────────────────────────────────────
# CUSTOMER QUERIES
# ─────────────────────────────────────────────────────────────

CUSTOMER_LIFETIME_VALUE = """
    -- CLV-Basis: Gesamtumsatz + Kaufhäufigkeit je Kunde + Segment
    -- Grundlage für RFM-Analyse in Python (SQL liefert Rohdaten)
    SELECT
        s.customer_id,
        c.segment,
        c.region,
        c.acquisition_channel,
        COUNT(*)            AS total_orders,
        SUM(s.revenue)      AS total_revenue,
        AVG(s.revenue)      AS avg_order_value,
        MIN(s.date)         AS first_purchase,
        MAX(s.date)         AS last_purchase,
        -- Tage zwischen erstem und letztem Kauf
        CAST(
            julianday(MAX(s.date)) - julianday(MIN(s.date))
        AS INTEGER)         AS customer_lifespan_days
    FROM sales s
    JOIN customers c ON s.customer_id = c.id
    GROUP BY s.customer_id
    ORDER BY total_revenue DESC
"""

ACQUISITION_CHANNEL_EFFICIENCY = """
    -- Akquisitionskanal-Analyse: Welcher Kanal bringt wertvollste Kunden?
    -- Kritische Business-Frage für Marketing-Budget-Allokation
    SELECT
        c.acquisition_channel,
        COUNT(DISTINCT c.id)        AS total_customers,
        AVG(cust_stats.total_revenue) AS avg_clv,
        AVG(cust_stats.total_orders)  AS avg_orders_per_customer,
        SUM(cust_stats.total_revenue) AS total_revenue_from_channel
    FROM customers c
    JOIN (
        SELECT
            customer_id,
            COUNT(*)       AS total_orders,
            SUM(revenue)   AS total_revenue
        FROM sales
        GROUP BY customer_id
    ) cust_stats ON c.id = cust_stats.customer_id
    GROUP BY c.acquisition_channel
    ORDER BY avg_clv DESC
"""

RAW_SALES_FOR_ANALYSIS = """
    -- Vollständiger Datensatz für Python-Analysen (RFM, Moving Average etc.)
    SELECT
        s.id,
        s.date,
        s.customer_id,
        s.product_id,
        s.quantity,
        s.revenue,
        s.region,
        s.channel,
        p.category,
        p.name          AS product_name,
        p.price,
        p.cost,
        c.segment,
        c.acquisition_channel
    FROM sales s
    JOIN products  p ON s.product_id  = p.id
    JOIN customers c ON s.customer_id = c.id
    ORDER BY s.date
"""

# ─────────────────────────────────────────────────────────────
# KPI SUMMARY QUERY
# ─────────────────────────────────────────────────────────────

KPI_OVERVIEW = """
    SELECT
        ROUND(SUM(revenue), 2)                              AS total_revenue,
        COUNT(*)                                            AS total_orders,
        COUNT(DISTINCT customer_id)                         AS active_customers,
        ROUND(AVG(revenue), 2)                              AS avg_order_value,
        ROUND(SUM(quantity), 0)                             AS total_units_sold,
        -- Gesamtmarge auf DB-Ebene (Revenue - Cost)
        ROUND(SUM(s.revenue - (s.quantity * p.cost)), 2)    AS gross_profit,
        ROUND(
            SUM(s.revenue - (s.quantity * p.cost)) / SUM(s.revenue) * 100,
            1
        )                                                   AS gross_margin_pct
    FROM sales s
    JOIN products p ON s.product_id = p.id
"""
