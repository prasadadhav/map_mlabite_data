# MLABiTe → PSA BESSER DB mapping (visual guide)

This guide describes how folders/files under `data/**` are mapped into the PSA BESSER SQLite schema using the **refactored accumulator**.

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

### Meaning of the path parts (refactor)

| Path part | Meaning in DB |
|---|---|
| `<project>` (e.g., `experiments`, `mistral`, `microcreditagent`) | **Project** (`project.name`) |
| `<test-name>` (e.g., `test-ageism`) | **MetricCategory** name = `ageism` (strip `test-`) |
| `<YYYYMMDD_HHMMSS>` | **Observation time** (one Observation per timestamp folder) |
| `<lang>` (e.g., `en_us`) | Dimension element linked to the Evaluation |

If no `<project>` can be inferred from the manifest, we use: `default_project`.

---

## 2) Core entities and relationships

### The big picture

```
Project
  └── Evaluation (config grouping)
        ├── Configuration + ConfParams
        ├── linked Elements (language, category, model(s), templates, etc.)
        ├── Observation(s)  ← each timestamp_dir is an observation (a run execution)
        │      └── Measure(s)  ← row/global/response values as measures
        └── MetricCategory (category registry + optional metric links)
```

### What is an **Evaluation** now?

An Evaluation represents a *stable configuration* within:

- Project
- MetricCategory
- Language
- Model set (from `config.json.aiModels`)
- Config signature (sha1 of canonical config.json)

So: **same config + same models + same language + same category** ⇒ same Evaluation, even if you ran it multiple times (different timestamp folders).

### What is an **Observation** now?

Observation = **one execution**, i.e. the `YYYYMMDD_HHMMSS` directory.

All outputs from that execution (row-level, global, and response text) are stored as **Measures** on that Observation, each tagged with its own **measurand Element**.

---

## 3) File → table mapping

### 3.1 Project

- **Table:** `project`
- **Source:** `provider_family` from manifest (derived from directory under `data/`)

```
project.id   = stable_int("project::<project_name>")
project.name = <project_name>
```

### 3.2 Tool

- **Table:** `tool`
- **Source:** constant (one tool row)

```
tool = (name="MLABiTe", version="NA", licensing="Open_Source")
```

### 3.3 MetricCategory

- **Table:** `metriccategory`
- **Source:** `<test-name>` mapped to category name (strip `test-`)

```
metriccategory.id   = stable_int("metriccategory::<category>")
metriccategory.name = <category>
```

Optionally:
- **Table:** `metriccategory_metric`
- We link the known core metrics used by MLABiTe to every encountered category.

### 3.4 Configuration + ConfParam

- **Tables:** `configuration`, `confparam`
- **Source:** `config.json`

We compute:

```
cfg_sig = sha1(canonical_json(config.json))[:16]
configuration.id = stable_int("configuration::<project>::<category>::<cfg_sig>")
```

For each top-level key in `config.json`:

```
confparam.id    = stable_int("confparam::<cfg_sig>::<key>")
confparam.conf_id = configuration.id
confparam.value = JSON string or raw string
```

### 3.5 Model (+ model registry dataset)

- **Tables:** `element`, `model`, `dataset`
- **Source:** `config.json.aiModels[]`

Each model pid becomes an `element(type_spec="model")` and a `model` row (polymorphic id).

### 3.6 Dataset

- **Tables:** `element`, `dataset`
- **Source:** templates in `*_evaluations.csv` + 2 fixed datasets

We always create:

- `__NO_TEMPLATE__` (fallback)
- `__RUN_CONTEXT__` (used as `observation.dataset_id` because one Observation can contain measures that come from many templates)

For each unique `Template` in `*_evaluations.csv` we also create a dataset row with `dataset.source = "<project>/<category>"`.

### 3.7 Evaluation + dimension links (both directions)

- **Table:** `evaluation`
- **Source:** grouping key = `(project, category, language, model_set, cfg_sig)`

We also create dimension **Element** rows and connect them via **two association tables**:

- `evaluation_element` (evaluation → element)
- `evaluates_eval` (element → evaluation)  ✅ this is the “mirror” so traversals work from either side

Linked elements include:

- `Language=<lang>`
- `MetricCategory=<category>`
- each `Model=<pid>`
- each `Template=<template>`
- plus per-run discovered dimensions like concerns / input types / reflection types / instances

### 3.8 Observation (timestamp folder)

- **Table:** `observation`
- **Source:** `<YYYYMMDD_HHMMSS>` directory name

```
observation.id = stable_int("observation::run::<project>::<category>::<cfg_sig>::<ts>::<lang>")
observation.whenObserved = parsed datetime from ts
observation.eval_id = evaluation.id
observation.tool_id = tool.id
observation.dataset_id = dataset("__RUN_CONTEXT__")
```

### 3.9 Measures (the actual numbers + response strings)

- **Table:** `measure`
- **Sources:** the 3 CSVs

#### a) `*_evaluations.csv`

For **each row**:
- Create a measurand Element `EvalRow=<hash>` describing the slice:
  - template, concern, input_type, reflection_type, row index
- Create 3 measures attached to the run Observation:
  - `evaluation`
  - `oracle_prediction`
  - `oracle_evaluation`

#### b) `*_global_evaluation.csv`

For **each row**:
- Create measurand Element `GlobalRow=<hash>`
- Create measures for:
  - Passed Nr, Failed Nr, Error Nr, Passed Pct, Failed Pct, Total, Tolerance, Tolerance Evaluation

#### c) `*_responses.csv`

For **each instance**:
- Create measurand Element `Instance=<instance>`
- Create measure:
  - `response_text` (string, up to 10k)

---

## 4) Notes on “features” and other empty tables

MLABiTe outputs (as provided) don’t naturally correspond to the DB’s `features` table (which is used by other tools / pipelines).
So **it is expected** for `features` to remain empty for this tool unless you later decide to model any of the “dimensions” (Concern / Input Type / Reflection Type / Template / Instance) specifically as “features” in your schema.

---

## 5) Loader ordering reminder

Your loading sequence stays the same overall (parent tables first), but you may add:

- `metriccategory.csv`
- `metriccategory_metric.csv`
- `evaluates_eval.csv`

to the sequence after their parents exist.

Your existing instructions document is still the reference for the general load order. fileciteturn0file5
