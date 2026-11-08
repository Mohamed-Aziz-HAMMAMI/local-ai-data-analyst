"""Local AI Data Analyser: Streamlit + Ollama + DuckDB (text-to-SQL).

Run with:  streamlit run main.py
Put the four CSV files in the same folder as this file.
"""

import html
import json
import re
import tempfile
import threading
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import sqlparse
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

st.set_page_config(page_title="Local AI Data Analyser", layout="wide")

def apply_ui_theme():
    """UI-only styling for the dark modern dashboard theme."""
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="stAppViewContainer"] {
                background: #0F1117;
            }
            .block-container {
                padding-top: 1.2rem !important;
                padding-bottom: 2.5rem !important;
                max-width: 1400px !important;
            }
            .stApp {
                background: #0F1117;
                color: #F5F7FF;
            }
            [data-testid="stSidebar"] {
                background: rgba(20, 23, 31, 0.95);
                border-right: 1px solid #2A2E3F;
            }
            .app-shell {
                margin-top: 0.5rem;
            }
            .topbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                padding: 1.25rem 1.5rem 1rem;
                margin-bottom: 1.2rem;
                border: 1px solid #2A2E3F;
                border-radius: 12px;
                background: rgba(26, 29, 41, 0.82);
                box-shadow: 0 6px 18px rgba(10, 12, 18, 0.18);
            }
            .title-wrap {
                display: flex;
                flex-direction: column;
                gap: 0.15rem;
            }
            .app-title {
                margin: 0;
                color: #F5F7FF;
                font-size: 2.1rem;
                font-weight: 700;
                letter-spacing: -0.04em;
            }
            .subtitle {
                margin: 0;
                color: #B6BED6;
                font-size: 0.95rem;
            }
            .badge-row {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.55rem;
            }
            .app-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.45rem 0.7rem;
                border-radius: 999px;
                border: 1px solid #2A2E3F;
                background: rgba(108, 92, 231, 0.12);
                color: #D8D2FF;
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.02em;
            }
            [data-testid="stVerticalBlockBorderWrapper"] {
                padding: 1rem;
                border: 1px solid #2A2E3F;
                border-radius: 12px;
                background: #1A1D29;
            }
            .empty-state {
                padding: 2rem;
                text-align: center;
                border: 1px dashed #3B4364;
                border-radius: 12px;
                background: rgba(26, 29, 41, 0.65);
                color: #D5DAEC;
            }
            .empty-state h3 {
                margin: 0 0 0.75rem;
                color: #F5F7FF;
                font-size: 1.45rem;
            }
            .empty-state p {
                margin: 0 0 1rem;
                color: #B6BED6;
            }
            .question-shell {
                padding: 1rem 1rem 0.25rem;
                background: rgba(26, 29, 41, 0.85);
                border: 1px solid #2A2E3F;
                border-radius: 12px;
                box-shadow: 0 12px 24px rgba(8, 10, 15, 0.16);
            }
            .stTextInput > div > div > input {
                background: rgba(15, 17, 23, 0.9);
                color: #F5F7FF;
                border: 1px solid #2A2E3F !important;
                border-radius: 12px !important;
                min-height: 3.3rem;
                padding: 0.7rem 1rem;
                font-size: 1.05rem;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
            }
            .stButton > button {
                background: linear-gradient(135deg, #6C5CE7 0%, #7B6AEF 100%);
                color: #F6F4FF;
                border: none;
                border-radius: 12px;
                font-weight: 600;
                padding: 0.8rem 1.1rem;
                min-height: 3.3rem;
                box-shadow: 0 10px 24px rgba(108, 92, 231, 0.35);
            }
            .stButton > button:hover {
                filter: brightness(1.04);
            }
            .stButton > button:focus:not(:active) {
                box-shadow: 0 0 0 0.2rem rgba(108, 92, 231, 0.25);
            }
            .stTabs [role="tablist"] {
                gap: 0.5rem;
                margin: 1rem 0 0.8rem;
                padding: 0.35rem;
                border-radius: 12px;
                background: rgba(26, 29, 41, 0.72);
                border: 1px solid #2A2E3F;
            }
            .stTabs [role="tablist"] button {
                color: #D4DBF8;
                border-radius: 10px;
                padding: 0.6rem 1rem;
                font-weight: 600;
            }
            .stTabs [role="tablist"] button[aria-selected="true"] {
                background: rgba(108, 92, 231, 0.18);
                border: 1px solid rgba(108, 92, 231, 0.32);
                color: #FFFFFF;
            }
            .stDataFrame {
                border-radius: 12px;
                border: 1px solid #2A2E3F;
                overflow: hidden;
            }
            .stMetric {
                border: 1px solid #2A2E3F;
                border-radius: 12px;
                background: rgba(26, 29, 41, 0.8);
                padding: 0.7rem 0.9rem;
            }
            .stMetric label {
                color: #B6BED6 !important;
            }
            .stMetric [data-testid="stMetricValue"] {
                color: #F5F7FF !important;
                font-size: 1.8rem !important;
            }
            .sidebar-section {
                margin-bottom: 1rem;
            }
            .sidebar-label {
                display: block;
                margin-bottom: 0.6rem;
                color: #D4DBF8;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .sidebar-table {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            .sidebar-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 0.75rem;
                padding: 0.5rem 0.65rem;
                border: 1px solid #2A2E3F;
                border-radius: 8px;
                background: rgba(18, 21, 30, 0.75);
                color: #E3EAFD;
                font-size: 0.9rem;
            }
            .sidebar-row span:last-child {
                color: #A4AEE8;
                font-weight: 600;
            }
            .sidebar-btn button {
                width: 100%;
                text-align: left !important;
                background: rgba(26, 29, 41, 0.92) !important;
                border: 1px solid #2A2E3F !important;
                color: #EAEAFB !important;
                border-radius: 10px !important;
                padding: 0.75rem 0.85rem !important;
                font-size: 0.9rem !important;
                margin-bottom: 0.5rem;
            }
            .sidebar-btn button:hover {
                border-color: rgba(108, 92, 231, 0.4) !important;
                background: rgba(108, 92, 231, 0.08) !important;
            }
            .answer-card {
                padding: 1.1rem 1.2rem;
                border-radius: 12px;
                background: rgba(26, 29, 41, 0.8);
                border: 1px solid #2A2E3F;
                color: #EEF3FF;
            }
            .answer-card strong {
                color: #FFFFFF;
            }
            .stCodeBlock code {
                color: #EAE8FF;
                background: #0B0D12;
            }
            .stCodeBlock {
                background: #0B0D12;
                border: 1px solid #2A2E3F;
                border-radius: 12px;
            }
            .stAlert {
                border-radius: 12px;
                border: 1px solid rgba(108, 92, 231, 0.4);
            }
            @media (max-width: 768px) {
                .topbar {
                    flex-direction: column;
                    align-items: flex-start;
                }
                .badge-row {
                    justify-content: flex-start;
                }
                .app-title {
                    font-size: 1.65rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_question(question):
    st.session_state["user_question"] = question
    st.session_state["auto_run"] = True


def render_sidebar(connection, tables, show_examples):
    with st.sidebar:
        st.markdown("<div class='sidebar-section'><span class='sidebar-label'>Model</span></div>", unsafe_allow_html=True)
        st.caption(f"{MODEL_NAME}")

        st.markdown("<div class='sidebar-section'><span class='sidebar-label'>Loaded tables</span></div>", unsafe_allow_html=True)
        for table_name in tables:
            row_count = connection.execute(f"SELECT COUNT(*) FROM {quote_ident(table_name)}").fetchone()[0]
            st.markdown(
                f"<div class='sidebar-row'><span>{table_name}</span><span>{row_count:,}</span></div>",
                unsafe_allow_html=True,
            )

        if show_examples:
            st.markdown("<div class='sidebar-section'><span class='sidebar-label'>Examples</span></div>", unsafe_allow_html=True)
            for idx, example in enumerate(EXAMPLE_QUESTIONS):
                button_key = f"example_{idx}"
                st.button(
                    example,
                    key=button_key,
                    width="stretch",
                    on_click=set_question,
                    args=(example,),
                )


def resolve_chart(df, spec):
    """Use the model's chart spec if valid; otherwise infer axes from the data."""
    kind, x, y = spec["chart_type"], spec["x"], spec["y"]
    spec_ok = (
        kind in {"bar", "line", "pie"}
        and x in df.columns
        and y in df.columns
        and x != y
        and pd.api.types.is_numeric_dtype(df[y])
    )
    if spec_ok:
        return kind, x, y
    if len(df) < 2:
        return "none", None, None

    numeric = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    labels = [column for column in df.columns if column not in numeric]
    if not numeric or not labels:
        return "none", None, None

    x, y = labels[0], numeric[0]
    looks_like_time = pd.api.types.is_datetime64_any_dtype(df[x]) or any(
        word in x.lower() for word in ("date", "month", "year")
    )
    return ("line" if looks_like_time else "bar"), x, y


def render_plotly_chart(df, spec):
    kind, x, y = resolve_chart(df, spec)
    if kind == "none":
        st.info("No chart is available for this result.")
        return

    plot_df = df[[x, y]].dropna().head(MAX_CHART_POINTS)
    if len(plot_df) < 2:
        st.info("There are not enough values to plot.")
        return

    if kind == "pie" and (len(plot_df) > 6 or (plot_df[y] < 0).any()):
        kind = "bar"

    title = spec.get("title") or ""
    if kind == "line":
        plot_df = plot_df.sort_values(x)
        fig = px.line(
            plot_df,
            x=x,
            y=y,
            title=title,
            template="plotly_dark",
            color_discrete_sequence=["#6C5CE7"],
        )
    elif kind == "pie":
        fig = px.pie(
            plot_df,
            names=plot_df[x].astype(str),
            values=y,
            title=title,
            template="plotly_dark",
            color_discrete_sequence=["#6C5CE7", "#7B6AEF", "#A399FF", "#C8C1FF", "#8E8AF6", "#4F47B7", "#E7E2FF", "#5E54D7"],
        )
    else:
        fig = px.bar(
            plot_df,
            x=x,
            y=y,
            title=title,
            template="plotly_dark",
            color_discrete_sequence=["#6C5CE7"],
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color="#F5F7FF"),
        legend=dict(font=dict(size=11), title=None),
        title=dict(font=dict(size=18, color="#F5F7FF")),
        xaxis=dict(showgrid=False, tickfont=dict(size=11), title_font=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=11), title_font=dict(size=12)),
    )
    st.plotly_chart(fig, width="stretch")


def render_metric_cards(df):
    if df.empty or len(df.columns) == 0:
        return

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        return

    metric_cols = numeric_cols[:4]
    metric_values = df.iloc[0][metric_cols].to_dict()
    cols = st.columns(len(metric_cols))
    for index, (key, value) in enumerate(metric_values.items()):
        with cols[index]:
            st.metric(
                label=key,
                value=f"{value:,.2f}" if pd.api.types.is_number(value) else str(value),
            )

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MODEL_NAME = "qwen2.5-coder:7b"
NUM_CTX = 8192  # Ollama's default (4096) is too small for the schema + rules
DATA_DIR = Path(__file__).resolve().parent

TABLES = {
    "order_items": "order_items.csv",
    "product_catalog": "product_catalog.csv",
    "sales_analytics": "ecommerce_sales_customer_analytics_150k.csv",
    "customer_master": "customer_master.csv",
}

EXAMPLE_QUESTIONS = [
    "Which 5 product categories generate the most profit?",
    "Show the monthly net sales for 2024.",
    "What is the return rate, and which sales channel has the highest average rating?",
    "What is the average customer acquisition cost for each sales channel?",
]

LOW_CARDINALITY_THRESHOLD = 20  # list distinct values for text columns up to this size
MAX_ROWS = 1000  # hard cap on rows returned by any query
MAX_CHART_POINTS = 30  # max categories/points drawn in a chart
QUERY_TIMEOUT_S = 30  # kill queries that run longer than this
MAX_RETRIES = 3  # how many times the model may repair a failing query

RELATIONSHIPS_TEXT = """
RELATIONSHIPS:
- order_items.order_id -> sales_analytics.order_id
- order_items.product_id -> product_catalog.product_id
- sales_analytics.customer_id -> customer_master.customer_id

NOTES:
- sales_analytics.return_status is NULL when the order was NOT returned (not missing data)
- sales_analytics.delivery_days is NULL when the order hasn't been delivered yet
- order_items.quantity is per line item; sales_analytics.quantity is the total per order
- customer_master and sales_analytics share 9 identical customer columns; prefer sales_analytics's copy to avoid unnecessary joins
"""

# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
SQL_TEMPLATE = """You are a SQL generator for DuckDB. Respond with ONLY SQL and NOTHING else.

Rules:
- Output only SQL. No explanations, no markdown, no code fences, no comments, no prefixes like "Here is the query".
- Use ONLY the tables and columns listed in the schema below. Never invent names.
- Write only SELECT queries (WITH ... SELECT is fine). Never modify data or the database.
- If the question cannot be answered with the schema, respond with exactly: INVALID_QUERY
- Answer only what is explicitly asked. If the user asks several separate questions, write one query per question and end EACH query with a semicolon.
- COUNT must always have an argument: use COUNT(*), never COUNT().
- Every non-aggregated column in the SELECT list must appear in the GROUP BY.
- If the needed columns are spread across several tables, JOIN them using exactly the relationships listed below. Never use a column from the wrong table.
- Every CASE WHEN must have a THEN: CASE WHEN condition THEN value ... END.
- To compare categories, GROUP BY that category and return one row per group.
- Give aggregates readable aliases and ROUND money and percentages to 2 decimals.
- Dates are DATE columns: use date_trunc('month', order_date), EXTRACT(year FROM order_date) or DATE '2024-01-01' literals.
- For "top N" questions use ORDER BY ... DESC LIMIT N.

EXAMPLES:

Question: Which 5 product categories generate the most profit?
SELECT pc.product_category, ROUND(SUM(oi.profit), 2) AS total_profit
FROM order_items oi
JOIN product_catalog pc ON oi.product_id = pc.product_id
GROUP BY pc.product_category
ORDER BY total_profit DESC
LIMIT 5;

Question: Show the monthly net sales for 2024.
SELECT date_trunc('month', order_date) AS month, ROUND(SUM(net_sales), 2) AS total_net_sales
FROM sales_analytics
WHERE order_date >= DATE '2024-01-01' AND order_date < DATE '2025-01-01'
GROUP BY month
ORDER BY month;

Question: What is the return rate, and which sales channel has the highest average rating?
SELECT ROUND(100.0 * COUNT(return_status) / COUNT(*), 2) AS return_rate_pct
FROM sales_analytics;

SELECT sales_channel, ROUND(AVG(customer_rating), 2) AS avg_rating
FROM sales_analytics
GROUP BY sales_channel
ORDER BY avg_rating DESC
LIMIT 1;

Question: What is the average customer acquisition cost for each sales channel?
SELECT sa.sales_channel, ROUND(AVG(cm.customer_acquisition_cost), 2) AS avg_acquisition_cost
FROM sales_analytics sa
JOIN customer_master cm ON sa.customer_id = cm.customer_id
GROUP BY sa.sales_channel
ORDER BY avg_acquisition_cost DESC;

SCHEMA:
{schema}

QUESTION:
{question}

SQL QUERY:
"""

FIX_TEMPLATE = """You are a SQL expert for DuckDB. The query below failed. Fix it and respond with ONLY the corrected SQL: one SELECT query, no explanations, no markdown, no comments.
Use ONLY the tables and columns listed in the schema. If the question truly cannot be answered with the schema, respond with exactly: INVALID_QUERY

SCHEMA:
{schema}

QUESTION:
{question}

FAILED QUERY:
{failed_sql}

ERROR:
{error}

CORRECTED SQL QUERY:
"""

CHART_TEMPLATE = """You explain query results in plain language and suggest how to chart them.

Respond with ONLY a JSON object, nothing else. Use exactly this shape:
{{
  "answer": "<one or two plain sentences answering the question, using the actual numbers>",
  "chart_type": "<one of: bar, line, pie, none>",
  "x": "<column name for the x-axis, or null>",
  "y": "<column name for the y-axis, or null>",
  "title": "<short chart title, or null>"
}}

Rules:
- Use only numbers that appear in the result below.
- If the result has a single row, set "chart_type" to "none".
- "x" and "y" must be exact column names from the result below. "y" must be a numeric column. Never invent column names.
- Use "pie" only for 5 or fewer categories. Otherwise prefer "bar".
- Use "line" only if the x-axis is a date or time column.

QUESTION:
{question}

TOTAL ROWS IN RESULT: {total_rows} (the CSV below shows at most the first 20)

RESULT (CSV):
{result_csv}

JSON OBJECT:
"""

FALLBACK_SPEC = {
    "answer": "Here are the results.",
    "chart_type": "none",
    "x": None,
    "y": None,
    "title": None,
}
CHART_TYPES = {"bar", "line", "pie", "none"}
READ_ONLY_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------
def quote_ident(name):
    return '"' + name.replace('"', '""') + '"'


def safe_table_name(filename, used):
    name = re.sub(r"\W", "_", Path(filename).stem).strip("_").lower()
    if not name:
        name = "table"
    if name[0].isdigit():
        name = f"t_{name}"

    unique_name = name
    suffix = 2
    while unique_name in used:
        unique_name = f"{name}_{suffix}"
        suffix += 1
    used.add(unique_name)
    return unique_name


def clean_column_names(con, table):
    """Rename columns to simple snake_case: 'departure.airport' -> 'departure_airport'."""
    old_names = [row[0] for row in con.execute(f"DESCRIBE {quote_ident(table)}").fetchall()]
    used, new_names = set(), []
    for old in old_names:
        base = re.sub(r"\W+", "_", old).strip("_").lower() or "column"
        if base[0].isdigit():
            base = "c_" + base
        name, i = base, 2
        while name in used:
            name, i = f"{base}_{i}", i + 1
        used.add(name)
        new_names.append(name)
    for old, new in zip(old_names, new_names):
        if old != new:
            con.execute(
                f"ALTER TABLE {quote_ident(table)} "
                f"RENAME COLUMN {quote_ident(old)} TO {quote_ident(new)}"
            )


def build_schema_text(con, table):
    """Describe one table for the LLM: column types, null counts and, for
    low-cardinality text/boolean columns, the list of possible values."""
    columns = con.execute(f"DESCRIBE {table}").fetchall()
    names = [col[0] for col in columns]
    counts = con.execute(
        "SELECT COUNT(*), " + ", ".join(f"COUNT({quote_ident(n)})" for n in names) + f" FROM {table}"
    ).fetchone()
    total, non_null_counts = counts[0], counts[1:]

    lines = [f"Table: {table} ({total} rows)"]
    for col, non_null in zip(columns, non_null_counts):
        name, dtype = col[0], col[1]
        line = f"  - {name} ({dtype})"
        nulls = total - non_null
        if nulls > 0:
            line += f" [nullable, {nulls} nulls]"
        if dtype in ("VARCHAR", "BOOLEAN"):
            values = [
                row[0]
                for row in con.execute(
                    f"SELECT DISTINCT {quote_ident(name)} FROM {table} "
                    f"WHERE {quote_ident(name)} IS NOT NULL LIMIT {LOW_CARDINALITY_THRESHOLD + 1}"
                ).fetchall()
            ]
            if len(values) <= LOW_CARDINALITY_THRESHOLD:
                line += f" -- values: {sorted(values, key=str)}"
        lines.append(line)
    return "\n".join(lines)


@st.cache_resource(show_spinner="Loading data into DuckDB...")
def init_db(signature, _sources):
    """Load the CSVs into in-memory DuckDB tables, build the schema text for
    the prompt, then lock the connection so generated SQL cannot touch files."""
    con = duckdb.connect()
    tables = []
    schema_parts = []
    if _sources:
        with tempfile.TemporaryDirectory() as temp_dir:
            for table, uploaded_file in _sources.items():
                csv_path = Path(temp_dir) / f"{table}.csv"
                try:
                    csv_path.write_bytes(uploaded_file.getvalue())
                    con.execute(
                        f"CREATE OR REPLACE TABLE {quote_ident(table)} AS "
                        "SELECT * FROM read_csv_auto(?, sample_size=-1)",
                        [str(csv_path)],
                    )
                    clean_column_names(con, table)
                    schema_parts.append(build_schema_text(con, quote_ident(table)))
                    tables.append(table)
                except Exception as error:
                    con.execute(f"DROP TABLE IF EXISTS {quote_ident(table)}")
                    st.error(f"Could not load {uploaded_file.name}: {error}")
        notes = "Tables may be joined on columns with matching names (e.g. customer_id)."
    else:
        for table, filename in TABLES.items():
            csv_path = (DATA_DIR / filename).as_posix()
            con.execute(
                f"CREATE OR REPLACE TABLE {quote_ident(table)} AS "
                "SELECT * FROM read_csv_auto(?, sample_size=-1)",
                [csv_path],
            )
            tables.append(table)

        date_range = con.execute(
            "SELECT MIN(order_date), MAX(order_date) FROM sales_analytics"
        ).fetchone()
        date_min, date_max = date_range if date_range is not None else (None, None)
        notes = RELATIONSHIPS_TEXT + f"- sales_analytics.order_date is a DATE column covering {date_min} to {date_max}\n"
        schema_parts = [build_schema_text(con, quote_ident(table)) for table in tables]

    schema = "\n\n".join(schema_parts) + "\n" + notes

    # Security: after loading, forbid file access and freeze the settings.
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con, schema, tables


# ----------------------------------------------------------------------------
# LLM chains
# ----------------------------------------------------------------------------
@st.cache_resource
def get_chains():
    llm = OllamaLLM(model=MODEL_NAME, temperature=0, num_ctx=NUM_CTX)
    # Request JSON-mode output for chart specifications.
    json_llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0,
        num_ctx=NUM_CTX,
        format="json",
    )
    return {
        "sql": PromptTemplate.from_template(SQL_TEMPLATE) | llm,
        "fix": PromptTemplate.from_template(FIX_TEMPLATE) | llm,
        "chart": PromptTemplate.from_template(CHART_TEMPLATE) | json_llm,
    }


# ----------------------------------------------------------------------------
# SQL helpers
# ----------------------------------------------------------------------------
def clean_sql(raw):
    """Strip markdown fences and stray 'sql' labels from the model output."""
    text = (raw or "").strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    text = text.replace("```", "").strip()
    text = re.sub(r"^sql\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def is_invalid_marker(text):
    return text.strip().upper().startswith("INVALID_QUERY")


def split_statements(raw):
    """Split the model output into separate SQL statements (on semicolons)."""
    statements = []
    for statement in sqlparse.split(clean_sql(raw)):
        statement = sqlparse.format(statement, strip_comments=True).strip().rstrip(";").strip()
        if statement:
            statements.append(statement)
    return statements


def validate_sql(sql):
    if not READ_ONLY_START.match(sql):
        raise ValueError("Only SELECT queries are allowed.")


def run_query(con, sql):
    """Run a validated query with a row cap and a timeout.

    The query is wrapped in a sub-select, which also guarantees it is
    read-only: INSERT / DROP / COPY etc. are syntax errors inside a subquery.
    """
    cursor = con.cursor()  # one cursor per run: DuckDB connections are not thread-safe
    timer = threading.Timer(QUERY_TIMEOUT_S, cursor.interrupt)
    timer.start()
    try:
        return cursor.execute(f"SELECT * FROM (\n{sql}\n) AS q LIMIT {MAX_ROWS}").df()
    except duckdb.InterruptException:
        raise TimeoutError(f"The query took longer than {QUERY_TIMEOUT_S} seconds and was stopped.")
    finally:
        timer.cancel()
        cursor.close()


def repair_missing_table_qualifier(con, sql, error):
    match = re.search(
        r'Referenced table "([^"]+)" not found!\s*Candidate tables: ([^\r\n]+)',
        str(error),
    )
    if not match:
        return sql

    missing_table, candidate_text = match.groups()
    candidates = re.findall(r'"([^"]+)"', candidate_text)
    if not candidates:
        return sql

    column_names = {}
    for candidate in candidates:
        try:
            column_names[candidate] = {
                row[0].casefold()
                for row in con.execute(f"DESCRIBE {quote_ident(candidate)}").fetchall()
            }
        except duckdb.Error:
            continue

    qualifier = rf'(?:"{re.escape(missing_table)}"|{re.escape(missing_table)})'
    reference = re.compile(
        rf'(?P<qualifier>{qualifier})\s*\.\s*'
        r'(?P<column>"(?:""|[^"])+?"|[A-Za-z_]\w*)',
        re.IGNORECASE,
    )

    def remove_unambiguous_qualifier(reference_match):
        column = reference_match.group("column").strip('"').replace('""', '"').casefold()
        matching_candidates = [
            candidate
            for candidate, columns in column_names.items()
            if column in columns
        ]
        if len(matching_candidates) == 1:
            return reference_match.group("column")
        return reference_match.group(0)

    return reference.sub(remove_unambiguous_qualifier, sql)


def generate_statements(chains, schema, question):
    raw = chains["sql"].invoke({"schema": schema, "question": question})
    if is_invalid_marker(clean_sql(raw)):
        return []
    return split_statements(raw)


def execute_with_repair(con, chains, schema, question, sql):
    """Run a query; if it fails, show the error to the model and retry."""
    error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            validate_sql(sql)
            return sql, run_query(con, sql), None
        except TimeoutError as timeout_error:
            return sql, None, timeout_error
        except Exception as run_error:
            error = run_error
            if attempt == MAX_RETRIES:
                break
            repaired_sql = repair_missing_table_qualifier(con, sql, run_error)
            if repaired_sql != sql:
                try:
                    return repaired_sql, run_query(con, repaired_sql), None
                except Exception as repaired_error:
                    error = repaired_error
            repaired = chains["fix"].invoke(
                {"schema": schema, "question": question, "failed_sql": sql, "error": str(error)}
            )
            if is_invalid_marker(clean_sql(repaired)):
                break
            statements = split_statements(repaired)
            if not statements:
                break
            sql = statements[0]
    return sql, None, error


# ----------------------------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------------------------
def parse_chart_spec(raw):
    """Find the first JSON object in the model output and keep the valid parts."""
    spec = dict(FALLBACK_SPEC)
    if not isinstance(raw, str) or not raw.strip():
        return spec

    decoder = json.JSONDecoder()
    for start, character in enumerate(raw):
        if character != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("answer") is not None and str(obj["answer"]).strip():
            spec["answer"] = str(obj["answer"]).strip()
        if obj.get("chart_type") in CHART_TYPES:
            spec["chart_type"] = obj["chart_type"]
        for key in ("x", "y", "title"):
            if isinstance(obj.get(key), str):
                spec[key] = obj[key]
        return spec
    return spec


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def answer_question(con, chains, schema, question):
    try:
        with st.spinner("Thinking..."):
            statements = generate_statements(chains, schema, question)
    except Exception as error:
        st.error(f"Could not get an answer from the language model. Is Ollama running? ({error})")
        return

    if not statements:
        st.warning(
            "I can't answer that with the available data. "
            "Try rephrasing it, or check the tables and columns in the sidebar."
        )
        return

    for number, statement in enumerate(statements, start=1):
        if len(statements) > 1:
            st.subheader(f"Query {number}")

        try:
            with st.spinner("Thinking..."):
                sql, result_df, error = execute_with_repair(con, chains, schema, question, statement)
        except Exception as llm_error:
            st.error(f"Could not get an answer from the language model. Is Ollama running? ({llm_error})")
            return

        if error is not None:
            st.error(f"Query failed: {error}")
            continue
        if result_df is None:
            st.error("Query returned no result.")
            continue
        if result_df.empty:
            st.info("The query returned no rows.")
            continue

        try:
            with st.spinner("Summarising results..."):
                raw_spec = chains["chart"].invoke(
                    {
                        "question": question,
                        "result_csv": result_df.head(20).to_csv(index=False),
                        "total_rows": len(result_df),
                    }
                )
            spec = parse_chart_spec(raw_spec)
        except Exception as summary_error:
            st.warning(f"The results are shown below, but the summary or chart failed: {summary_error}")
            spec = {
                "answer": "Here are the results.",
                "chart_type": "none",
                "x": None,
                "y": None,
                "title": None,
            }

        answer_text = html.escape(spec.get("answer", "Here are the results."))

        with st.container(border=True):
            answer_tab, chart_tab, data_tab, sql_tab = st.tabs(["Answer", "Chart", "Data", "SQL"])

            with answer_tab:
                st.markdown(f"<div class='answer-card'>{answer_text}</div>", unsafe_allow_html=True)
                if len(result_df) == 1:
                    render_metric_cards(result_df)

            with chart_tab:
                render_plotly_chart(result_df, spec)

            with data_tab:
                st.dataframe(result_df, width="stretch")
                if len(result_df) >= MAX_ROWS:
                    st.caption(f"Showing the first {MAX_ROWS} rows only.")

            with sql_tab:
                st.code(sql, language="sql")


apply_ui_theme()
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type="csv",
    accept_multiple_files=True,
)
upload_signature = tuple((file.name, file.size) for file in uploaded_files)
if st.session_state.get("_upload_signature") != upload_signature:
    st.session_state.pop("user_question", None)
    st.session_state.pop("auto_run", None)
    st.session_state["_upload_signature"] = upload_signature

used_table_names = set()
sources = {
    safe_table_name(uploaded_file.name, used_table_names): uploaded_file
    for uploaded_file in uploaded_files
}

st.markdown(
    """
    <div class="app-shell">
        <div class="topbar">
            <div class="title-wrap">
                <h1 class="app-title">Local AI Data Analyser</h1>
                <p class="subtitle">Ask questions about your data. Runs 100% locally.</p>
            </div>
            <div class="badge-row">
                <span class="app-badge">Ollama</span>
                <span class="app-badge">DuckDB</span>
                <span class="app-badge">Streamlit</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not sources and any(not (DATA_DIR / filename).is_file() for filename in TABLES.values()):
    st.info("Upload CSV files in the sidebar to get started.")
    st.stop()

connection, full_schema, tables = init_db(upload_signature, sources)
llm_chains = get_chains()
render_sidebar(connection, tables, show_examples=not sources)

if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""

with st.container(border=True):
    with st.form("ask_form"):
        user_question = st.text_input(
            "Your question",
            key="user_question",
            placeholder="Ask a question about your data",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze", width="stretch")

if not st.session_state.get("user_question") and not submitted:
    empty_state_text = (
        "Ask a question about your uploaded data."
        if sources
        else "Ask a business question in plain English and the app will translate it to SQL locally."
    )
    st.markdown(
        f"""
        <div class="empty-state">
            <h3>Ready to explore your data</h3>
            <p>{html.escape(empty_state_text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not sources:
        with st.container(border=True):
            example_columns = st.columns(2)
            for idx, example in enumerate(EXAMPLE_QUESTIONS):
                with example_columns[idx % len(example_columns)]:
                    st.button(
                        example,
                        key=f"empty_{idx}",
                        width="stretch",
                        on_click=set_question,
                        args=(example,),
                    )

if submitted or st.session_state.pop("auto_run", False):
    question = (st.session_state.get("user_question") or "").strip()
    if not question:
        st.warning("Please type a question first.")
    else:
        answer_question(connection, llm_chains, full_schema, question)