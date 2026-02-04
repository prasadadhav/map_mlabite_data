$DB = "mla_bite_jan_2026.db"
$CSV = "data\data_accumulated"
$SPEC = "spec_templates_all_tables_MLABite"

$tables = @(
  "project",
  "tool",
  "datashape",
  "element",
  "dataset",
  "model",
  "configuration",
  "confparam",
  "evaluation",
  "evaluation_element",
  "metric",
  "direct",
  "observation",
  "measure"
)

foreach ($t in $tables) {
  Write-Host "Loading $t ..."
  python .\csv_to_sql_loader.py `
    --db $DB `
    --csv "$CSV\$t.csv" `
    --spec "$SPEC\$t.yml"
}
