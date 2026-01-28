# CSV→SQL Spec Templates

Generated from database: ai_sandbox_PSA_16_Oct_2025.db

Each `.yml` and `.json` here is a mapping spec for a single table. Edit:
- `mode`: `"insert"` or `"upsert"` (requires a unique/PK `key`)
- `key`: list of columns used to match for upserts
- `columns`: map CSV column names to DB columns. Replace `<CSV_column>` with your headers.

Supported rules (YAML/JSON keys):
- `from`: CSV column name
- `const`: constant value for all rows
- `transform`: pipe-separated ops: strip|lower|upper|title
- `as_type`: int|float|str
- `expr`: Python expression with `row[...]`

Tips:
- For autoincrement integer PKs, prefer not to map them; let SQLite assign IDs.
- For foreign keys, ensure referenced IDs exist (load reference tables first).
- Loader warns if NOT NULL columns without defaults are missing in your mapping.

Use with the loader script:
    python csv_to_sql_loader.py --db path/to.db --csv results.csv --spec specs/<table>.yml
