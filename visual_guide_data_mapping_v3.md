# MLABiTe → PSA BESSER DB Mapping (v3.1)

This guide explains how raw MLABiTe result files are mapped into the PSA BESSER relational schema **to support reporting** like:

- Which **templates** failed?
- What was the **populated prompt** (template instance) for those failures?
- What were the **responses**?
- What were the **metric measures** (evaluation + oracle metrics) for that exact template-instance?

---

## Big idea

### ✅ Templates are Datasets
Each **prompt template** from `*_evaluations.csv` becomes a **Dataset**:

- `dataset.source` = **Template** (string with placeholders)
- `dataset.version` = **Language** (folder name: `en_us`, `fr_fr`, …)
- `dataset.id` = **element.id** (template Dataset is backed by an Element row)
- `element.name` (for the template Dataset element) = **string(dataset.id)**
- `dataset.datashape_id` points to a `datashape` representing the **context**:  
  `datashape.accepted_target_values = "<project>/<concern>"`

### ✅ Template instances are Elements
Each executed instance (populated/filled prompt) is captured as an **Element**:

- `element.name` = the **dataset_id** (points back to the template Dataset)
- `element.description` = the **template instance** (filled prompt)  
  (from `*_responses.csv` column `Instance`)

### ✅ Responses live in Observation.description
Each response to a template-instance is represented by an **Observation**:

- `observation.dataset_id` = template Dataset
- `observation.description` = response text (full)
- `measure.measurand_id` = template-instance Element
- measures attach per observation for evaluation/oracle metrics

---

## Relationship diagram

```
datashape ("project/concern")
        |
        v
dataset (template + language)
        |
        v
observation (timestamped run instance)
   |                    |
   |                    v
   |                measure(s)
   |              (evaluation/oracle)
   v
element (template instance / populated prompt)
```

Key join path for reporting:

```
dataset (template)
  -> observation (response)
  -> measure (metrics)
  -> element (populated prompt for that observation)
```

---

## Source files → Tables

### 1) `*_evaluations.csv`
Used for:
- `dataset.source` (Template)
- Category detection via `Concern`
- Per-row measures:
  - `Evaluation`
  - `Oracle Prediction`
  - `Oracle Evaluation`

### 2) `*_responses.csv`
Used for:
- `observation.description` (Response)
- `element.description` (Instance = populated prompt)

> Note: `*_responses.csv` does not include Template/Concern, so mapping is done by
> **row-order alignment per Model** within a (project, concern, language, timestamp) slice.

### 3) `*_global_evaluation.csv`
Used for run-level measures stored as a separate “RUN” observation.

### 4) `config.json`
Used for:
- `configuration` + `confparam`
- Evaluation grouping signature

### 5) Manifest (`manifest.json`)
Provides the run structure and file paths:
- project name comes from `provider_family`
- timestamp directories define observation time

---

## Table-by-table mapping

### project
- One row per `provider_family` (top-level directory under `data/`)
- `project.name = provider_family`
- `project.status = "Ready"`

---

### datashape
Purpose: store **context** that applies to a set of template datasets.

- `datashape.accepted_target_values = "<project>/<concern>"`
- `<concern>` is derived from CSV `Concern` column (source of truth)

Examples:
- `experiments/ageism`
- `mistral/religion`
- `microcreditagent/job_status`

---

### dataset (templates)
A dataset represents a template in a particular language and context.

- `dataset.source = Template` (from `*_evaluations.csv`)
- `dataset.version = language` (directory name)
- `dataset.datashape_id` = datashape(project/concern)
- `dataset_type = "Test"`, `licensing = "Proprietary"` (defaults; adjust if you want)

**Uniqueness:**
`(project/concern, language, template_hash)`.

---

### element (template instances / populated prompts)
A **template instance** element corresponds to the **filled** prompt that generated a response.

- `element.type_spec = "element"`
- `element.name = "<dataset_id>"`  ✅ (dataset reference)
- `element.description = "<Instance>"`  ✅ populated prompt (from responses CSV)

---

### observation (template instance response)
One Observation = one response for one model/template row at one timestamp.

- `observation.whenObserved` from timestamp directory (`YYYYMMDD_HHMMSS`)
- `observation.dataset_id` = template Dataset (template + language + context)
- `observation.description` = response text
- `observation.eval_id` links to evaluation group
- `observation.tool_id` = MLABiTe tool id

Also created:
- A run-level observation per (project, concern, language, timestamp) named `"<concern>:RUN"`
  storing global summary measures.

---

### metric / direct / derived
- Most metrics are `Direct`
- **Derived metrics (type_spec = "Derived")** in v3.1:
  - `Passed Pct`
  - `Failed Pct`
  - `Tolerance Evaluation`

For derived metrics:
- a row is created in `derived` with `expression = "NA"` (placeholder)

All other metrics used here remain:
- `evaluation`
- `oracle_prediction`
- `oracle_evaluation`
- `Passed Nr`, `Failed Nr`, `Error Nr`, `Total`, `Tolerance`

---

### measure
Per-row measures (attached to template-instance observation):
- `metric_id` in {evaluation, oracle_prediction, oracle_evaluation}
- `measurand_id` = template-instance element (populated prompt)
- `observation_id` = the observation holding the response

Run-level measures (attached to RUN observation):
- global counts and percentages from `*_global_evaluation.csv`

---

## “Which templates failed?” — query logic

You can answer the reporting questions with these joins:

### A) Templates that failed
- failures are `measure(metric='evaluation', value='Fail' or 'Error' etc)`
- get template text from `dataset.source`

Join chain:
1. `measure` → `observation` (by observation_id)
2. `observation` → `dataset` (by dataset_id)
3. `dataset.source` is the template

### B) The populated prompt that produced that failure
From the same failure measure row:
- `measure.measurand_id` → `element.description` (filled prompt)

### C) The response
- `observation.description`

### D) Related metrics for the same template-instance
- all `measure` rows with same `observation_id` (and same measurand_id if you want strictness)

---

## Not-null / NA conventions (v3.1)

- CSV reading uses `keep_default_na=False`, so literal `"NA"` stays `"NA"`.
- For required text columns, missing values are emitted as `"NA"` to satisfy NOT NULL.
- Response strings may be empty (still NOT NULL), but the script will emit `"NA"` only if truly missing.

---

## Notes / Limitations

### Response↔Template alignment
Because `*_responses.csv` does not contain the template, mapping uses:
- group by `Model`
- align row order with templates in `*_evaluations.csv` for the same concern slice

If the counts differ, the response may be blank for some template rows.

If you want perfect linking, the ideal upstream fix is:
- include `Template` and `Concern` columns in the responses CSV, or
- include a stable row id shared across evaluations/responses.

