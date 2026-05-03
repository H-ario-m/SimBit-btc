"""
db.py — Persistent storage layer for BTC Forecast Dashboard.

Strategy:
  - LOCAL: If no DATABASE_URL secret is set, uses local SQLite (predictions.db).
  - CLOUD: If DATABASE_URL is set (via Streamlit secrets or env), uses PostgreSQL.
           This allows deployment on Streamlit Community Cloud with Supabase for
           permanent, reboot-proof storage.

Setup for Streamlit Cloud:
  1. Create a free project at https://supabase.com
  2. Go to Settings -> Database -> Connection string -> URI (Transaction Pooler)
  3. In Streamlit Cloud app settings -> Secrets, add:
        DATABASE_URL = "postgresql://postgres.<ref>:<password>@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd
try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

DB_PATH = Path(__file__).parent / "predictions.db"


def _get_database_url() -> str | None:
    """Retrieve DATABASE_URL from Streamlit secrets or environment variable."""
    if _STREAMLIT_AVAILABLE:
        try:
            # Try flat key first
            if "DATABASE_URL" in st.secrets:
                return st.secrets["DATABASE_URL"]
            # Try nested [database] table style
            db_section = st.secrets.get("database", {})
            if isinstance(db_section, dict) and "DATABASE_URL" in db_section:
                return db_section["DATABASE_URL"]
        except Exception:
            pass
    return os.environ.get("DATABASE_URL", None)


_PG_WORKING: bool | None = None  # None = untested, True = ok, False = failed


def _is_postgres() -> bool:
    """Return True only if DATABASE_URL is set AND a connection is reachable."""
    global _PG_WORKING
    url = _get_database_url()
    if not url or not _PG_AVAILABLE:
        return False
    if _PG_WORKING is not None:
        return _PG_WORKING
    # Test the connection once at startup
    try:
        conn = psycopg2.connect(url, connect_timeout=8)
        conn.close()
        _PG_WORKING = True
    except Exception as e:
        import sys
        print(f"[db] PostgreSQL connection FAILED, falling back to SQLite: {e}", file=sys.stderr)
        _PG_WORKING = False
    return _PG_WORKING


@contextmanager
def _get_pg_conn() -> Generator:
    url = _get_database_url()
    conn = psycopg2.connect(url, connect_timeout=8)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _get_sqlite_conn() -> Generator:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the database schema if it doesn't exist."""
    if _is_postgres():
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        fetched_at TEXT,
                        target_time TEXT,
                        low_95 FLOAT,
                        high_95 FLOAT,
                        current_price FLOAT,
                        actual_price FLOAT,
                        profile TEXT
                    )
                    """
                )
    else:
        with _get_sqlite_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetched_at TEXT,
                    target_time TEXT,
                    low_95 REAL,
                    high_95 REAL,
                    current_price REAL,
                    actual_price REAL,
                    profile TEXT
                )
                """
            )
            cursor = conn.execute("PRAGMA table_info(predictions)")
            cols = [row["name"] for row in cursor.fetchall()]
            if "profile" not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN profile TEXT DEFAULT 'unknown'")


def _normalize_time(time_str: str) -> str:
    """Standardize time strings to a consistent ISO format without timezone offset for string matching."""
    try:
        dt = pd.to_datetime(time_str)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return time_str


def save_prediction(
    fetched_at: str,
    target_time: str,
    low_95: float,
    high_95: float,
    current_price: float,
    profile: str = "precision",
) -> None:
    """Save a new live prediction to the database."""
    n_fetched = _normalize_time(fetched_at)
    n_target = _normalize_time(target_time)
    print(f"[db] Saving prediction for target_time: {n_target}")
    
    if _is_postgres():
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions (fetched_at, target_time, low_95, high_95, current_price, profile)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (n_fetched, n_target, low_95, high_95, current_price, profile),
                )
    else:
        with _get_sqlite_conn() as conn:
            conn.execute(
                """
                INSERT INTO predictions (fetched_at, target_time, low_95, high_95, current_price, profile)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (n_fetched, n_target, low_95, high_95, current_price, profile),
            )


def update_actual_price(target_time: str, actual_price: float) -> None:
    """Update the actual price for a target_time (fills NULL rows only)."""
    n_target = _normalize_time(target_time)
    print(f"[db] Attempting to update actual price for target_time: {n_target} to {actual_price}")
    
    if _is_postgres():
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                # Use timestamp casting for robust matching in PostgreSQL
                cur.execute(
                    """
                    UPDATE predictions
                    SET actual_price = %s
                    WHERE (target_time = %s OR target_time::timestamp = %s::timestamp)
                      AND actual_price IS NULL
                    """,
                    (actual_price, n_target, n_target),
                )
                count = cur.rowcount
                print(f"[db] Postgres update matched {count} rows.")
    else:
        with _get_sqlite_conn() as conn:
            cur = conn.execute(
                """
                UPDATE predictions
                SET actual_price = ?
                WHERE target_time = ? AND actual_price IS NULL
                """,
                (actual_price, n_target),
            )
            print(f"[db] SQLite update matched {cur.rowcount} rows.")


def get_recent_predictions(limit: int = 100) -> list[dict[str, Any]]:
    """Retrieve recent predictions from the database."""
    if _is_postgres():
        with _get_pg_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM predictions ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                return [dict(row) for row in cur.fetchall()]
    else:
        with _get_sqlite_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_prediction_history_df(limit: int = 50) -> pd.DataFrame:
    """Retrieve history as a pandas DataFrame with hit/miss calculated."""
    preds = get_recent_predictions(limit)
    if not preds:
        return pd.DataFrame()
    
    df = pd.DataFrame(preds)
    if "actual_price" in df.columns:
        # Calculate Hit/Miss for rows that have an actual price
        def check_hit(row):
            if pd.isna(row["actual_price"]) or row["actual_price"] == 0:
                return "Pending"
            is_hit = row["low_95"] <= row["actual_price"] <= row["high_95"]
            return "HIT" if is_hit else "MISS"
            
        df["Result"] = df.apply(check_hit, axis=1)
        
    return df
