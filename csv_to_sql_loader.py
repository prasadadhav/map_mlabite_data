"""
How to run:
python313 csv_to_sql_loader.py \
  --db ./ai_sandbox_PSA_16_Oct_2025.db \
  --csv /path/to/your_results.csv \
  --spec ./spec_template.yml

  

python313 .\csv_to_sql_loader.py `
  --db ".\ai_sandbox_PSA_16_Oct_2025_MLABite.db" `
  --csv ".\MLABiTe\20250704_160557\en_us\OpenAIGPT35Turbo\20250704160636_global_evaluation.csv" `
  --spec ".\spec_templates_all_tables_MLABite\model.yml"

  
python313 .\csv_to_sql_loader.py `
  --db ".\ai_sandbox_PSA_16_Oct_2025_MLABite.db" `
  --csv ".\MLABiTe\20250704_160557\en_us\OpenAIGPT35Turbo\20250704160636_global_evaluation.csv" `
  --spec ".\spec_templates_all_tables_MLABite\configuration.yml"
"""


"""
Generic CSV → SQL Loader for arbitrary LLM/ML evaluation outputs.

Usage outline:
1) Create a mapping spec (YAML or JSON) describing how CSV columns map to DB columns.
2) Run: python csv_to_sql_loader.py --db /path/to.db --csv /path/to.csv --spec /path/to.yml --table observations
3) Supports constants, renaming, simple transforms, and insert/upsert.

The script introspects required columns from the DB and validates presence.
"""
import argparse
import sqlite3
import pandas as pd
import json
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None

# --- Helpers -----------------------------------------------------------------

def load_spec(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Use JSON spec or `pip install pyyaml`.")
        return yaml.safe_load(text)
    return json.loads(text)

def get_table_schema(conn, table: str):
    info = pd.read_sql_query(f"PRAGMA table_info('{table}')", conn)
    fks  = pd.read_sql_query(f"PRAGMA foreign_key_list('{table}')", conn)
    return info, fks

def required_cols(table_info: pd.DataFrame):
    # Required = NOT NULL without a DEFAULT and not part of an autoincrement PK (heuristic)
    req = []
    for _, r in table_info.iterrows():
        if r["notnull"] == 1 and r["dflt_value"] is None:
            # If it's a PK and type is INTEGER, we assume autoincrement (common pattern) and skip
            if r["pk"] == 1 and "INT" in (r["type"] or "").upper():
                continue
            req.append(r["name"])
    return req

def apply_mapping(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    Spec format (example):
    {
      "table": "observations",
      "mode": "insert",   # or "upsert"
      "key": ["name"],    # required for upsert
      "columns": {
        "db_col_A": {"from": "csv_col_a"},                # rename
        "db_col_B": {"const": "LLM-v1"},                  # constant
        "db_col_C": {"from": "csv_col_c", "transform": "strip|lower"},  # simple transforms
        "db_col_D": {"from": "score", "as_type": "float"},             # type cast
        "db_col_E": {"expr": "row['a'] + '-' + row['b']"}              # Python eval expression with 'row' dict
      }
    }
    """
    out = pd.DataFrame(index=df.index)
    for db_col, rule in spec.get("columns", {}).items():
        if "from" in rule:
            series = df[rule["from"]]
        elif "const" in rule:
            series = pd.Series([rule["const"]] * len(df), index=df.index)
        elif "expr" in rule:
            # WARNING: eval of user-supplied expressions. Use with care in your environment.
            series = df.apply(lambda row: eval(rule["expr"]), axis=1)
        else:
            raise ValueError(f"Column rule for '{db_col}' must include 'from', 'const', or 'expr'.")

        # transforms
        t = rule.get("transform")
        if t:
            for op in t.split("|"):
                op = op.strip().lower()
                if op == "strip":
                    series = series.astype(str).str.strip()
                elif op == "lower":
                    series = series.astype(str).str.lower()
                elif op == "upper":
                    series = series.astype(str).str.upper()
                elif op == "title":
                    series = series.astype(str).str.title()
                else:
                    raise ValueError(f"Unknown transform: {op}")

        # type cast
        as_type = rule.get("as_type")
        if as_type:
            if as_type == "int":
                series = pd.to_numeric(series, errors="coerce").astype("Int64")
            elif as_type == "float":
                series = pd.to_numeric(series, errors="coerce").astype(float)
            elif as_type == "str":
                series = series.astype(str)
            else:
                raise ValueError(f"Unsupported as_type: {as_type}")

        out[db_col] = series

    # Keep only columns defined in the spec
    return out

def upsert_dataframe(conn, df: pd.DataFrame, table: str, key_cols: list[str]):
    """
    Generic upsert using SQLite's ON CONFLICT clause.
    Assumes a UNIQUE constraint or PRIMARY KEY exists on key_cols.
    """
    if df.empty:
        return 0

    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join([f'"{c}"' for c in cols])

    update_set = ", ".join([f'"{c}"=excluded."{c}"' for c in cols if c not in key_cols])
    key_list   = ", ".join([f'"{c}"' for c in key_cols])

    # PSA
    # sql = f"""
    # INSERT INTO "{table}" ({col_list})
    # VALUES ({placeholders})
    # ON CONFLICT({key_list}) DO UPDATE SET
    #   {update_set};
    # """
    if not update_set.strip():
        # Nothing to update (table only has key cols) -> DO NOTHING
        sql = f"""
        INSERT INTO "{table}" ({col_list})
        VALUES ({placeholders})
        ON CONFLICT({key_list}) DO NOTHING;
        """
    else:
        sql = f"""
        INSERT INTO "{table}" ({col_list})
        VALUES ({placeholders})
        ON CONFLICT({key_list}) DO UPDATE SET
        {update_set};
        """

    cur = conn.cursor()
    cur.executemany(sql, df[cols].where(pd.notnull(df), None).values.tolist())
    conn.commit()
    return cur.rowcount

def insert_dataframe(conn, df: pd.DataFrame, table: str):
    if df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join([f'"{c}"' for c in cols])
    sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders});'
    cur = conn.cursor()
    cur.executemany(sql, df[cols].where(pd.notnull(df), None).values.tolist())
    conn.commit()
    return cur.rowcount

# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path, help="Path to SQLite .db")
    ap.add_argument("--csv", required=True, type=Path, help="Path to input CSV")
    ap.add_argument("--spec", required=True, type=Path, help="Path to mapping spec (YAML or JSON)")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    table = spec["table"]
    mode  = spec.get("mode", "insert")
    key   = spec.get("key", [])

    conn = sqlite3.connect(args.db.as_posix())
    tbl_info, _ = get_table_schema(conn, table)

    # Load CSV
    df = pd.read_csv(args.csv)

    # Apply mapping
    mapped = apply_mapping(df, spec)

    # Validate required columns
    req = required_cols(tbl_info)
    missing_required = [c for c in req if c not in mapped.columns]
    if missing_required:
        print(f"[WARN] Mapped data missing NOT NULL columns without defaults: {missing_required}")

    if mode == "upsert":
        if not key:
            raise ValueError("Upsert mode requires 'key' in spec.")
        count = upsert_dataframe(conn, mapped, table, key)
    else:
        count = insert_dataframe(conn, mapped, table)

    print(f"Loaded {count} rows into '{table}'.")

if __name__ == "__main__":
    main()
