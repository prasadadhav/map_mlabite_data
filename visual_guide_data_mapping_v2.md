# MLABiTe → PSA BESSER DB mapping (visual guide) — v2 (Concern-aware categories)

This guide describes how folders/files under `data/**` are mapped into the PSA BESSER SQLite schema using the **refactored accumulator v2**, where **MetricCategory is derived from the CSV `Concern` column** (because a single run can include multiple concerns).

---

## 1) How we interpret your results folder

A **run** is discovered the same way as before (timestamp folder + CSVs + config.json). fileciteturn0file3

### A run on disk (typical)

```
data/<project>/<test-name>/<YYYYMMDD_HHMMSS>/<lang>/
  config.json
  *_evaluations.csv
  *_global_evaluation.csv
  *_responses.csv
```

### Meaning of path parts

| Path part | Meaning in DB |
|---|---|
| `<project>` (e.g., `experiments`, `mistral`, `microcreditagent`) | **Project** (`project.name`) |
| `<test-name>` (e.g., `test-ageism`) | **Fallback only** for category (used only if `Concern` is missing) |
| `<YYYYMMDD_HHMMSS>` | **Observation time** (one Observation per timestamp *per category*) |
| `<lang>` (e.g., `en_us`) | Dimension element linked to the Evaluation |

If no `<project>` can be inferred from the manifest, we use: `default_project`.

---

## 2) Core entities and relationships

### Big picture (schema-level)

```
Project
  └── Evaluation  (stable grouping)
        ├── Configuration + ConfParams
        ├── linked Elements (language, category, model(s), templates, etc.)
        ├── Observation(s)  ← each (timestamp_dir, language, category) becomes one Observation
        │      └── Measure(s)  ← row/global/response values as measures
        └── MetricCategory (registry + optional category↔metric links)
```

### The key v2 change (multi-category runs)

A **single timestamp run** can contain multiple bias tests at once:

```
timestamp_dir = 20250728101351
evaluations.csv has Concern values:
  ageism
  religion
  xenophobia
  job status
```

So we do this:

```
for each unique Concern in evaluations.csv/global_evaluation.csv:
    MetricCategory = normalize(Concern)
    make Evaluation(category-specific)
    make Observation(category-specific)
    attach only that Concern's rows as Measures
    duplicate response_text measures into each category observation
```

This is why folder names like `test-ageism/` are **not** trusted as category.

---

## 3) What is an Evaluation vs Observation now?

### Evaluation (stable config grouping)

Evaluation represents a *stable configuration*, grouped by:

- Project
- **MetricCategory** (from `Concern`)
- Language
- Model set (from `config.json.aiModels`)
- Config signature (sha1 of canonical `config.json`)

So: **same config + same models + same language + same category** ⇒ same Evaluation, even if you ran it multiple times.

### Observation (one execution per category)

Observation = a *single execution instance*, but **scoped to one category**:

- One timestamp directory can become **multiple observations**:
  - Observation(ageism)
  - Observation(religion)
  - Observation(xenophobia)
  - …

Each Observation:
- points to its Evaluation
- stores Measures for that category’s row/global metrics
- also stores duplicated response_text (because responses.csv has no Concern)

---

## 4) File → table mapping

Below is a balanced “visual + text” mapping.

### 4.1 Project

**Table:** `project`  
**Source:** `provider_family` in manifest (derived from the top-level directory under `data/`).

```
project.id   = stable_int("project::<project_name>")
project.name = <project_name>
```

### 4.2 Tool

**Table:** `tool`  
**Source:** constant (one row).

```
tool = (name="MLABiTe", version="NA", licensing="Open_Source")
```

### 4.3 MetricCategory (from Concern)

**Table:** `metriccategory`  
**Source:** `Concern` column values in:
- `*_evaluations.csv`
- `*_global_evaluation.csv`

Normalization:
- trim → lowercase → spaces/hyphens to `_` → remove punctuation  
Example: `"Job Status"` → `job_status`

```
metriccategory.id   = stable_int("metriccategory::<category>")
metriccategory.name = <category>
```

Optional:
- **Table:** `metriccategory_metric`
- We link “known core metrics” to each category for category browsing.

### 4.4 Configuration + ConfParam

**Tables:** `configuration`, `confparam`  
**Source:** `config.json`

```
cfg_sig = sha1(canonical_json(config.json))[:16]
configuration.id = stable_int("configuration::<project>::<category>::<cfg_sig>")
```

Each top-level key becomes a confparam row.

### 4.5 Evaluation + dimension links (bidirectional)

**Table:** `evaluation`  
**Grouped by:** (project, category, language, model_set, cfg_sig)

We also create dimension Elements and link them in BOTH directions:

```
evaluation_element: (eval -> element)
evaluates_eval:    (element -> eval)   ✅ mirror edges for reverse traversal
```

Dimensions linked include:
- Language
- MetricCategory
- Models
- Template (if present)
- Concern/InputType/ReflectionType (from row slices)
- Instance (for responses)

### 4.6 Observation (timestamp + category)

**Table:** `observation`  
**Source:** `timestamp_dir`, but **one per category**.

```
observation.id = stable_int("observation::run::<project>::<category>::<cfg_sig>::<ts>::<lang>")
observation.whenObserved = parsed datetime from ts
observation.eval_id = evaluation.id
observation.tool_id = tool.id
observation.dataset_id = dataset("__RUN_CONTEXT__")
```

### 4.7 Measures (numbers + response strings)

**Table:** `measure`  
**Sources:** the 3 CSVs

#### a) `*_evaluations.csv` (category-sliced)

For each row **whose Concern normalizes to this category**:
- Create a measurand Element `EvalRow=<hash>` that encodes:
  - template, raw concern, input_type, reflection_type, row index
- Measures created:
  - `evaluation`
  - `oracle_prediction`
  - `oracle_evaluation`

Visual:

```
evaluations.csv row
   ├─ measurand Element (EvalRow=...)
   ├─ Measure(metric=evaluation)
   ├─ Measure(metric=oracle_prediction)
   └─ Measure(metric=oracle_evaluation)
(all attached to Observation(category))
```

#### b) `*_global_evaluation.csv` (category-sliced)

For each row **whose Concern normalizes to this category**:
- measurand Element `GlobalRow=<hash>`
- Measures created:
  - Passed Nr, Failed Nr, Error Nr, Passed Pct, Failed Pct, Total, Tolerance, Tolerance Evaluation

#### c) `*_responses.csv` (duplicated across categories)

responses.csv does **not** have Concern. So we:

- Create measurand Element `Instance=<instance>`
- Create `response_text` measure
- Attach that same measure to **each category’s Observation** for that run

Visual:

```
responses.csv instance row
  └─ response_text measure
        ↳ duplicated into Observation(ageism)
        ↳ duplicated into Observation(religion)
        ↳ duplicated into Observation(xenophobia)
        ...
```

This matches your note that duplication may be necessary when the same raw data belongs in multiple categories.

---

## 5) What this fixes (the 2 pain points)

### ✅ Folder name no longer drives category
Even if the folder is `test-ageism/`, the category can still become `job_status` if that’s what Concern says.

### ✅ Reverse traversal works
Because we always write both:
- `evaluation_element`
- `evaluates_eval`

So tracing from Element → Evaluation works, not only Evaluation → Element.

---

## 6) Loader ordering reminder

Your existing loading order remains the reference. fileciteturn1file1

If you weren’t loading them before, add:
- `metriccategory.csv`
- `metriccategory_metric.csv`
- `evaluates_eval.csv`

after their parent tables exist.
