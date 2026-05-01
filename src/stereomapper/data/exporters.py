"""Export helpers for results outputs."""

import sqlite3

import pandas as pd


RESULT_TABLES = (
    "clusters",
    "relationships",
    "cluster_members",
    "relationship_members",
)


def export_results_workbook(results_db_path: str, workbook_path: str) -> None:
    """Export core results tables to a single workbook with one sheet per table."""
    with sqlite3.connect(results_db_path, timeout=30.0) as con:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for table_name in RESULT_TABLES:
                frame = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
                frame.to_excel(writer, sheet_name=table_name, index=False)
