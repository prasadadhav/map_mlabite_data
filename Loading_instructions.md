<!-- ------------------------------------------- -->
# The 4 pipeline stages (high-level explaination)
<!-- ------------------------------------------- -->

## Stage A: Discover the runs

Goal: build a normalized list of runs regardless of directory nesting.
- Input: `data/**`
- Output: manifest.json

Rule: a “run” is any directory named `YYYYMMDD_HHMMSS` that contains language subdirs and those contain `*_evaluations.csv`, `*_global_evaluation.csv`, `*_responses.csv`, plus `config.json`

## Stage B: Normalize each run into “entities”

For each (`timestamp_dir`, `language`) pair we extract:

- tool: `(name="MLABiTe", version=timestamp_dir, licensing=Open_Source)`
-project: recommended `project.name = test_name` (e.g., `test-ageism`) + optional “suite project”
- model: from `config.json aiModels[]` (usually 1 model), then enrich using registry
- dataset: comes from `*_evaluations.csv` template column (per your instruction)
- evaluation: one per (`timestamp_dir`, `language`, `model`) (or per language if you prefer)

## Stage C: Metrics vs Measures

the results csv file are:
- `*_evaluations.csv` columns include: Provider, Model, Concern, Language, Input Type, Reflection Type, Template, Oracle Evaluation, Oracle Prediction, Evaluation
- `*_global_evaluation.csv` includes totals/pcts/tolerance
- `*_responses.csv` includes Provider, Model, Instance, Response

So we can define metrics like:

1. instance_eval (per row in `evaluations.csv`)
    * measures: `evaluation` (pass/fail/etc), `oracle_prediction`, `oracle_evaluatio`n, plus dimension values (concern, input_type, reflection_type, template_id)

2. global_eval (per row in `global_evaluation.csv`)
    * measures: passed_nr, failed_nr, error_nr, passed_pct, failed_pct, total, tolerance, tolerance_evaluation

3. response_text (per row in responses.csv)
    * measure: response (string up to 10k)
    * plus instance id

> Important: dimensions like Concern/Input Type/Reflection Type should be treated as attributes (or “features/elements”) of the observation, not separate evaluations.


## Stage D: Generate accumulated CSVs with FK-guarding

Goal: write table CSVs into: `data/data_accumulated/<table>.csv`

Key point: FK safety comes from deterministic IDs, not from directory separation.

So we generate IDs via UUIDv5 (stable across runs/machines), e.g.:
- project_id = uuid5(NS, f"project::{test_name}")
- tool_id = uuid5(NS, f"tool::MLABiTe::{timestamp_dir}")
- model_id = uuid5(NS, f"model::{pid}")
- evaluation_id = uuid5(NS, f"eval::{timestamp_dir}::{language}::{pid}")
- metric_id = uuid5(NS, f"metric::{metric_name}")
- observation_id = uuid5(NS, f"obs::{evaluation_id}::{instance_or_rowhash}")
- measure_id = uuid5(NS, f"measure::{observation_id}::{measure_name}")

This guarantees:

- re-running doesn’t duplicate rows
- multiple runs accumulate safely
- you can delete/refresh one run if needed (because IDs are predictable)

<!-- ------------------------------------------- -->
# Code
<!-- ------------------------------------------- -->

## first set the python313 env alias
```powershell
Set-Alias python313 "C:\Users\adhav\AppData\Local\Programs\Python\Python313\python.exe"
```

## to generate the SQL DB run the py script generated from the BESSER web platform
```powershell
python313 .\sql_alchemy.py
```


## generate the db
```powershell
python313 .\discover_run.py
```

## accumulate the data into csvs
old
```powershell
python313 .\accumulate_to_data_accumulated.py
```

refactored
```powershell
python313 accumulate_to_data_accumulated_refactored_v2.py --repo . --manifest manifest.json --out data/data_accumulated
```




## Core tables (minimum viable)

- `project.csv`
- `tool.csv`
- `configuration.csv`
- `confparam.csv`
- `evaluation.csv`
- `element.csv`
- `dataset.csv`
- `model.csv`
- `metric.csv`
- `direct.csv`
- `observation.csv`
- `measure.csv`
- `evaluation_element.csv`







<!-- ------------------------------------------- -->
# Loading sequence
<!-- ------------------------------------------- -->

## Project
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\project.csv `
  --spec spec_templates_all_tables_MLABite\project.yml
```

## Tools
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\tool.csv `
  --spec spec_templates_all_tables_MLABite\tool.yml
```


## Datashape
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\datashape.csv `
  --spec spec_templates_all_tables_MLABite\datashape.yml
```


## Element (⚠️ critical parent table)
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\element.csv `
  --spec spec_templates_all_tables_MLABite\element.yml
```


## Dataset
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\dataset.csv `
  --spec spec_templates_all_tables_MLABite\dataset.yml
```



## Model
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\model.csv `
  --spec spec_templates_all_tables_MLABite\model.yml
```


## Configuration
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\configuration.csv `
  --spec spec_templates_all_tables_MLABite\configuration.yml
```


## ConfParam
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\confparam.csv `
  --spec spec_templates_all_tables_MLABite\confparam.yml
```



## Evaluation
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\evaluation.csv `
  --spec spec_templates_all_tables_MLABite\evaluation.yml
```


## Evaluation ↔ Element (dimension linking)
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\evaluation_element.csv `
  --spec spec_templates_all_tables_MLABite\evaluation_element.yml
```



## Metric
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\metric.csv `
  --spec spec_templates_all_tables_MLABite\metric.yml
```


## Direct (metric subtype)
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\direct.csv `
  --spec spec_templates_all_tables_MLABite\direct.yml
```


## Observation
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\observation.csv `
  --spec spec_templates_all_tables_MLABite\observation.yml
```


## Measure (final leaf table)
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\measure.csv `
  --spec spec_templates_all_tables_MLABite\measure.yml
```


```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\measure_drop_na.csv `
  --spec spec_templates_all_tables_MLABite\measure.yml
```

```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\measure_modded.csv `
  --spec spec_templates_all_tables_MLABite\measure.yml
```

## 
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\comments.csv `
  --spec spec_templates_all_tables_MLABite\comments.yml
```


## 
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\metriccategory_metric.csv `
  --spec spec_templates_all_tables_MLABite\metriccategory_metric.yml
```


## 
```powershell
python313 .\csv_to_sql_loader.py `
  --db mla_bite_feb_2026_refactor.db `
  --csv data\data_accumulated\metriccategory.csv `
  --spec spec_templates_all_tables_MLABite\metriccategory.yml
```


## 
```powershell

```


## 
```powershell

```


## 
```powershell

```

## 
```powershell

```