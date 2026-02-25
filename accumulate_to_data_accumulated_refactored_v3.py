#!/usr/bin/env python3
from __future__ import annotations

"""
Accumulator — v3.1 (Template-centric datasets + template→response instance mapping)

Changes vs v3:
- DO NOT store responses as a measure (responses live in Observation.description).
- Respect NOT NULL constraints by emitting explicit "NA" for missing required text fields.
  (We keep_default_na=False when reading CSVs so literal "NA" is preserved.)
- The *instance* (populated template / filled prompt) is stored in ELEMENT.description
  for the per-template-instance element. ELEMENT.name is set to the corresponding Dataset ID.
- Metric.type_spec:
    - "Passed Pct", "Failed Pct", "Tolerance Evaluation" => Derived
    - all other metrics => Direct

Core v3 behavior retained:
- Dataset.source = prompt Template (from *_evaluations.csv)
- Dataset.version = language (directory)
- Datashape.accepted_target_values = "<project>/<concern>" (concern from CSV "Concern")
- One Observation per template instance response (timestamped execution):
    Observation.dataset_id -> the template Dataset
    Observation.description -> response text (truncated)
- Measures attach to the same observation (evaluation/oracle_* plus global metrics on run obs).

Constraint:
- *_responses.csv lacks Template/Concern. Responses are aligned to templates by per-model row order
  within a (project, concern, language, timestamp) slice. If counts mismatch, response may be blank.

This script writes CSVs into data/data_accumulated (or --out), deduping by stable keys.
"""

import json
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd

# ----------------------------
# Configuration / constants
# ----------------------------

NA_STR = "NA"  # Use explicit NA placeholder for required NOT NULL text fields

DEFAULT_OBSERVER = "MLABiTe"

TOOL_NAME = "MLABiTe"
TOOL_SOURCE = "LIST"
TOOL_LICENSING = "Open_Source"

PROJECT_STATUS = "Ready"
EVALUATION_STATUS = "Done"

DATASET_LICENSING_DEFAULT = "Proprietary"
DATASET_TYPE_DEFAULT = "Test"

MODEL_DATA_DEFAULT = NA_STR

MEASURE_ERROR_DEFAULT = "Not measured"
MEASURE_UNIT_DEFAULT = "Not defined"
MEASURE_UNCERTAINTY_DEFAULT = 0.0

# Hard safety caps (align with your upgraded DB schema; keep conservative caps here)
MEASURE_VALUE_MAXLEN = 10000
OBS_DESC_MAXLEN = 10000
ELEMENT_DESC_MAXLEN = 10000
DATASET_SOURCE_MAXLEN = 10000  # if you widened it; if not, loader/spec should truncate

DEFAULT_PROJECT_NAME = "default_project"

# ----------------------------
# Utilities
# ----------------------------

def stable_int(key: str) -> int:
    """Deterministic int ID from a key (32-bit friendly)."""
    h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    n = int(h[:8], 16)
    return (n % 2_000_000_000) + 1

def parse_timestamp_dir(ts: str) -> str:
    """Convert YYYYMMDD_HHMMSS -> ISO datetime string for whenObserved."""
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.isoformat(sep=" ")
    except Exception:
        return datetime.utcnow().isoformat(sep=" ")

def sniff_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    return ";" if sample.count(";") > sample.count(",") else ","

def read_csv_flex(path: Path) -> pd.DataFrame:
    """Robust CSV reader (delimiter sniff + tolerant fallback)."""
    sep = sniff_sep(path)
    try:
        return pd.read_csv(path, sep=sep, engine="c", keep_default_na=False)
    except Exception:
        try:
            return pd.read_csv(
                path,
                sep=sep,
                engine="python",
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
            )
        except Exception:
            alt = "," if sep == ";" else ";"
            return pd.read_csv(
                path,
                sep=alt,
                engine="python",
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
            )

def read_responses_csv_robust(path: Path) -> pd.DataFrame:
    """
    Responses can contain separators/newlines in text. Parse line-by-line and treat
    everything after the 3rd separator as Response.
    Expected columns: Provider, Model, Instance, Response
    """
    sep = sniff_sep(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return pd.DataFrame(columns=["Provider", "Model", "Instance", "Response"])

    rows = []
    for raw in lines[1:]:
        if not raw.strip():
            continue
        parts = raw.split(sep)
        if len(parts) < 2:
            continue
        if len(parts) >= 4:
            provider = parts[0].strip()
            model = parts[1].strip()
            instance = parts[2].strip()
            response = sep.join(parts[3:]).strip()
        else:
            provider = parts[0].strip() if len(parts) > 0 else ""
            model = parts[1].strip() if len(parts) > 1 else ""
            instance = parts[2].strip() if len(parts) > 2 else ""
            response = ""
        rows.append({"Provider": provider, "Model": model, "Instance": instance, "Response": response})
    return pd.DataFrame(rows, columns=["Provider", "Model", "Instance", "Response"])

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def upsert_append(out_csv: Path, df: pd.DataFrame, key_cols: List[str]) -> None:
    """Append + dedupe by key cols."""
    if out_csv.exists():
        prev = read_csv_flex(out_csv)
        merged = pd.concat([prev, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="first")
        merged.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False)

def norm_path(p: str) -> Path:
    return Path(p.replace("\\", os.sep))

def safe_str_required(x: Any) -> str:
    """For NOT NULL text fields: return NA_STR when missing."""
    if x is None:
        return NA_STR
    if isinstance(x, float) and pd.isna(x):
        return NA_STR
    s = str(x)
    # If the source is truly empty, keep it empty ONLY when meaningful (e.g., response can be empty).
    # For required descriptive fields we prefer NA_STR.
    if s == "":
        return NA_STR
    return s

def safe_str_optional(x: Any) -> str:
    """For optional text fields where empty string is acceptable."""
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)

def normalize_metric_category_from_test_name(test_name: str) -> str:
    """Fallback: map 'test-ageism' -> 'ageism'."""
    t = (test_name or "").strip()
    if t.lower().startswith("test-"):
        return t[5:]
    return t or "unknown_category"

def normalize_metric_category_from_concern(concern: str) -> str:
    """Primary: derive metric category from CSV Concern column."""
    c = (concern or "").strip()
    if not c:
        return ""
    c = c.lower().replace("&", " and ")
    c = re.sub(r"[\s\-]+", "_", c)
    c = re.sub(r"[^a-z0-9_]+", "", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def config_signature(cfg: dict) -> str:
    return hashlib.sha1(canonical_json(cfg).encode("utf-8", errors="ignore")).hexdigest()[:16]

def template_key(template: str) -> str:
    """Stable short key for a template (avoid huge IDs)."""
    t = (template or "").strip()
    if not t:
        return "unknown_template"
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()[:16]

# ----------------------------
# Model registry (starter)
# ----------------------------

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "OpenAIGPT35Turbo": {"canonical": "gpt-35-turbo", "source": "OpenAI", "licensing": "Proprietary"},
    "OpenAIGPT4o": {"canonical": "gpt-4o", "source": "OpenAI", "licensing": "Proprietary"},
    "OpenAIGPT4oMini": {"canonical": "gpt-4o-mini", "source": "OpenAI", "licensing": "Proprietary"},
    "MistralMedium": {"canonical": "mistral-medium-2505", "source": "Mistral", "licensing": "Proprietary"},
    "MistralLarge": {"canonical": "mistral-large", "source": "Mistral", "licensing": "Proprietary"},
    "ClaudeSonnet": {"canonical": "claude-sonnet", "source": "Anthropic", "licensing": "Proprietary"},
    "GeminiPro": {"canonical": "gemini-pro", "source": "Google", "licensing": "Proprietary"},
    "Llama3": {"canonical": "llama-3", "source": "Meta", "licensing": "Open_Source"},
    "DeepSeekR1": {"canonical": "deepseek-r1", "source": "DeepSeek", "licensing": "Open_Source"},
    "Phi": {"canonical": "phi", "source": "Microsoft", "licensing": "Open_Source"},
    "Grok": {"canonical": "grok", "source": "xAI", "licensing": "Proprietary"},
    "MicroCreditAssistScore": {"canonical": "microcredit-assist-score", "source": "Creditum AI SARL", "licensing": "Proprietary"},
}

def lookup_model(pid: str) -> Dict[str, str]:
    if pid in MODEL_REGISTRY:
        return MODEL_REGISTRY[pid]
    compact = re.sub(r"[^a-zA-Z0-9]+", "", pid).lower()
    for k, v in MODEL_REGISTRY.items():
        if re.sub(r"[^a-zA-Z0-9]+", "", k).lower() == compact:
            return v
    src = "Unknown"
    lic = "Proprietary"
    if any(x in compact for x in ["llama", "phi", "deepseek"]):
        lic = "Open_Source"
    return {"canonical": pid, "source": src, "licensing": lic}

# ----------------------------
# Main accumulator
# ----------------------------

def main(repo_root: Path, manifest_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)

    manifest_full_path = (repo_root / "data" / "data_accumulated" / manifest_path)
    manifest = json.loads(manifest_full_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])

    rows: Dict[str, List[Dict[str, Any]]] = {t: [] for t in [
        "project",
        "tool",
        "configuration",
        "confparam",
        "datashape",
        "element",
        "dataset",
        "model",
        "metriccategory",
        "metriccategory_metric",
        "evaluation",
        "evaluation_element",
        "evaluates_eval",
        "metric",
        "direct",
        "derived",
        "observation",
        "measure",
    ]}

    # Tool row (single)
    tool_id = stable_int(f"tool::{TOOL_NAME}")
    rows["tool"].append({"id": tool_id, "source": TOOL_SOURCE, "version": "vX.x.x", "name": TOOL_NAME, "licensing": TOOL_LICENSING})

    # Shared model registry dataset element (for model.dataset_id FK)
    model_registry_dataset_element_id = stable_int("element::dataset::ModelRegistry")
    rows["element"].append({
        "id": model_registry_dataset_element_id,
        "project_id": None,  # nullable
        "type_spec": "dataset",
        "name": "ModelRegistry",
        "description": "Dataset placeholder for model registry linkage"
    })
    ds_modelshape_id = stable_int("datashape::model_registry")
    rows["datashape"].append({"id": ds_modelshape_id, "accepted_target_values": "ModelRegistry"})
    rows["dataset"].append({
        "id": model_registry_dataset_element_id,
        "source": "ModelRegistry",
        "version": "vX.x.x",
        "licensing": "Open_Source",
        "dataset_type": "Test",
        "datashape_id": ds_modelshape_id,
    })

    # Metrics registry
    metric_ids: Dict[str, int] = {}

    DERIVED_METRIC_NAMES = {"Passed Pct", "Failed Pct", "Tolerance Evaluation"}

    def ensure_metric(metric_name: str, description: str = "") -> int:
        key = metric_name.strip()
        if key in metric_ids:
            return metric_ids[key]
        mid = stable_int(f"metric::{key}")
        metric_ids[key] = mid

        if key in DERIVED_METRIC_NAMES:
            rows["metric"].append({"id": mid, "type_spec": "Derived", "name": key, "description": (description or key)[:100]})
            # Expression is required; we keep it explicit but minimal unless you later want real formulas.
            rows["derived"].append({"id": mid, "expression": NA_STR})
        else:
            rows["metric"].append({"id": mid, "type_spec": "Direct", "name": key, "description": (description or key)[:100]})
            rows["direct"].append({"id": mid})

        return mid

    # Essential metrics (row-level)
    ensure_metric("evaluation", "Row-level evaluation result")
    ensure_metric("oracle_prediction", "Row-level oracle prediction")
    ensure_metric("oracle_evaluation", "Row-level oracle evaluation")

    # Global metrics
    for m in ["Passed Nr", "Failed Nr", "Error Nr", "Passed Pct", "Failed Pct", "Total", "Tolerance", "Tolerance Evaluation"]:
        ensure_metric(m, f"Global metric {m}")

    # Element helper
    element_seen: Set[int] = set()

    def ensure_element(type_spec: str, name: str, description: str, project_id: Optional[int]) -> int:
        eid = stable_int(f"element::{type_spec}::{name}::{hashlib.sha1((description or '').encode('utf-8', errors='ignore')).hexdigest()[:10]}")
        if eid in element_seen:
            return eid
        element_seen.add(eid)
        rows["element"].append({
            "id": eid,
            "project_id": project_id,
            "type_spec": safe_str_required(type_spec)[:50],
            "name": safe_str_required(name)[:100],
            "description": (safe_str_required(description)[:ELEMENT_DESC_MAXLEN]),
        })
        return eid

    # Datashape per project/concern
    datashape_ids: Dict[str, int] = {}

    def ensure_datashape(project_name: str, metric_category: str) -> int:
        key = f"{project_name}/{metric_category}"
        if key in datashape_ids:
            return datashape_ids[key]
        ds_id = stable_int(f"datashape::{key}")
        datashape_ids[key] = ds_id
        rows["datashape"].append({"id": ds_id, "accepted_target_values": safe_str_required(key)[:100]})
        return ds_id

    # Dataset per (project/concern, language, template)
    dataset_seen: Set[int] = set()

    def ensure_template_dataset(project_id: int, project_name: str, metric_category: str, language: str, template_text: str) -> int:
        """
        Create/reuse a Dataset representing a prompt template within (project/concern, language).
        Dataset.id == Element.id (type_spec='dataset').
        """
        ds_id = ensure_datashape(project_name, metric_category)
        tkey = template_key(template_text)
        elem_id = stable_int(f"element::dataset::{project_name}/{metric_category}::{language}::{tkey}")

        if elem_id not in dataset_seen:
            dataset_seen.add(elem_id)
            element_seen.add(elem_id)
            template_clean = (template_text or NA_STR).strip()
            rows["element"].append({
                "id": elem_id,
                "project_id": project_id,
                "type_spec": "dataset",
                "name": str(elem_id),
                "description": (template_clean[:5000] or NA_STR),
            })
            rows["dataset"].append({
                "id": elem_id,
                "source": (template_clean[:DATASET_SOURCE_MAXLEN] or NA_STR),
                "version": safe_str_required(language)[:100],
                "licensing": DATASET_LICENSING_DEFAULT,
                "dataset_type": DATASET_TYPE_DEFAULT,
                "datashape_id": ds_id,
            })
        return elem_id

    # Model rows
    model_seen: Set[int] = set()

    def ensure_model(pid: str, project_id: int) -> int:
        model_elem_id = stable_int(f"element::model::{pid}")
        if model_elem_id in model_seen:
            return model_elem_id
        model_seen.add(model_elem_id)
        info = lookup_model(pid)
        rows["element"].append({
            "id": model_elem_id,
            "project_id": project_id,
            "type_spec": "model",
            "name": safe_str_required(pid)[:100],
            "description": safe_str_required(f"{info.get('source','Unknown')} | {info.get('canonical',pid)}")[:ELEMENT_DESC_MAXLEN],
        })
        rows["model"].append({
            "id": model_elem_id,
            "pid": safe_str_required(pid)[:100],
            "data": safe_str_required(info.get("canonical", pid) or MODEL_DATA_DEFAULT)[:100],
            "source": safe_str_required(info.get("source", "Unknown"))[:100],
            "licensing": safe_str_required(info.get("licensing", "Proprietary"))[:11],
            "dataset_id": model_registry_dataset_element_id,
        })
        return model_elem_id

    # MetricCategory registry (browsing/filtering)
    metric_category_ids: Dict[str, int] = {}

    def ensure_metric_category(name: str, derived_from: str) -> int:
        n = (name.strip() or "unknown_category")[:100]
        if n in metric_category_ids:
            return metric_category_ids[n]
        mc_id = stable_int(f"metriccategory::{n}")
        metric_category_ids[n] = mc_id
        rows["metriccategory"].append({
            "id": mc_id,
            "name": n,
            "description": safe_str_required(f"Metric category '{n}' derived from {derived_from}")[:100],
        })
        return mc_id

    # Configuration + params
    config_registry: Dict[str, int] = {}

    def get_or_create_configuration(project_name: str, metric_category: str, cfg: dict) -> Tuple[int, str]:
        sig = config_signature(cfg)
        key = f"{project_name}/{metric_category}/{sig}"
        if key in config_registry:
            return config_registry[key], sig
        config_id = stable_int(f"configuration::{key}")
        config_registry[key] = config_id
        rows["configuration"].append({
            "id": config_id,
            "name": safe_str_required(f"config_{project_name}_{metric_category}_{sig}")[:100],
            "description": safe_str_required(f"Config for {project_name}/{metric_category} (sig={sig})")[:100],
        })
        for k, v in cfg.items():
            cp_id = stable_int(f"confparam::{key}::{k}")
            rows["confparam"].append({
                "id": cp_id,
                "param_type": "json",
                "value": json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v,
                "conf_id": config_id,
                "name": safe_str_required(k)[:100],
                "description": "from config.json",
            })
        return config_id, sig

    # Evaluation grouping
    eval_registry: Dict[Tuple[str, str, str, str, str], int] = {}

    def get_or_create_evaluation(
        project_id: int,
        project_name: str,
        metric_category: str,
        language: str,
        model_pids: List[str],
        config_id: int,
        cfg_sig: str
    ) -> int:
        models_key = "|".join(sorted([m.strip() for m in model_pids if m.strip()])) or "UnknownModel"
        k = (project_name, metric_category, language, models_key, cfg_sig)
        if k in eval_registry:
            return eval_registry[k]
        eval_id = stable_int(f"evaluation::{project_name}::{metric_category}::{language}::{models_key}::{cfg_sig}")
        eval_registry[k] = eval_id
        rows["evaluation"].append({"id": eval_id, "status": EVALUATION_STATUS, "config_id": config_id, "project_id": project_id})

        def link_eval_element(eid: int) -> None:
            rows["evaluation_element"].append({"ref": eid, "eval": eval_id})
            rows["evaluates_eval"].append({"evaluates": eid, "evalu": eval_id})

        # Language element
        lang_eid = ensure_element("element", f"Language={language}", "Language dimension", project_id)
        link_eval_element(lang_eid)

        # MetricCategory element
        cat_eid = ensure_element("element", f"MetricCategory={metric_category}", "Metric category dimension", project_id)
        link_eval_element(cat_eid)

        # Models
        for pid in sorted(set([p for p in model_pids if p.strip()])):
            mid = ensure_model(pid, project_id)
            link_eval_element(mid)

        return eval_id

    # ----------------------------
    # Iterate runs
    # ----------------------------

    for run in runs:
        ts = run["timestamp_dir"]
        test_name = run.get("test_name") or "unknown_test"
        language = run.get("language") or "unknown_lang"
        paths = run["paths"]

        project_name = run.get("provider_family") or DEFAULT_PROJECT_NAME
        project_id = stable_int(f"project::{project_name}")
        rows["project"].append({"id": project_id, "name": safe_str_required(project_name)[:100], "status": PROJECT_STATUS})

        # Read CSVs
        evals_df = read_csv_flex(repo_root / norm_path(paths["evals_csv"]))
        global_df = read_csv_flex(repo_root / norm_path(paths["global_csv"]))

        resp_path = repo_root / norm_path(paths["responses_csv"])
        try:
            resp_df = read_csv_flex(resp_path)
        except Exception:
            resp_df = read_responses_csv_robust(resp_path)
        if "Response" not in resp_df.columns or "Model" not in resp_df.columns:
            resp_df = read_responses_csv_robust(resp_path)

        # Load config.json
        cfg_path = repo_root / norm_path(paths["config_json"])
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        # Determine models
        model_pids = cfg.get("aiModels") or []
        if not model_pids:
            if "Model" in evals_df.columns and len(evals_df) > 0:
                model_pids = [safe_str_optional(evals_df["Model"].iloc[0])]
            else:
                model_pids = ["UnknownModel"]

        # Categories present (from Concern)
        fallback_category = normalize_metric_category_from_test_name(test_name)
        categories: Set[str] = set()
        for df in (evals_df, global_df):
            if "Concern" in df.columns:
                for v in df["Concern"]:
                    cat = normalize_metric_category_from_concern(safe_str_optional(v))
                    if cat:
                        categories.add(cat)
        if not categories:
            categories = {fallback_category}

        # Prepare response lists per model (for alignment)
        resp_by_model: Dict[str, pd.DataFrame] = {}
        if len(resp_df) > 0 and "Model" in resp_df.columns:
            for m, g in resp_df.groupby("Model", dropna=False):
                g2 = g.copy().reset_index(drop=True)
                g2["_ord"] = range(len(g2))
                resp_by_model[safe_str_optional(m)] = g2

        when_obs = parse_timestamp_dir(ts)

        # For each concern/category: evaluation group & template-instance observations
        for metric_category in sorted(categories):
            derived_from = "Concern column" if metric_category != fallback_category else "Concern column (or folder fallback)"
            mc_id = ensure_metric_category(metric_category, derived_from=derived_from)

            config_id, cfg_sig = get_or_create_configuration(project_name, metric_category, cfg)

            eval_id = get_or_create_evaluation(
                project_id=project_id,
                project_name=project_name,
                metric_category=metric_category,
                language=language,
                model_pids=model_pids,
                config_id=config_id,
                cfg_sig=cfg_sig
            )

            # Optional: category↔metric browsing links
            for metric_name, mid in metric_ids.items():
                mm_id = stable_int(f"metriccategory_metric::{mc_id}::{mid}")
                rows["metriccategory_metric"].append({"id": mm_id, "metriccategory_id": mc_id, "metric_id": mid})

            # Slice evals/global by concern
            if "Concern" in evals_df.columns:
                eval_slice = evals_df[evals_df["Concern"].astype(str).apply(lambda x: normalize_metric_category_from_concern(x) == metric_category)].copy()
            else:
                eval_slice = evals_df.copy()

            if "Concern" in global_df.columns:
                global_slice = global_df[global_df["Concern"].astype(str).apply(lambda x: normalize_metric_category_from_concern(x) == metric_category)].copy()
            else:
                global_slice = global_df.copy()

            # Ordered templates per model in this category slice
            tmpl_rows_by_model: Dict[str, pd.DataFrame] = {}
            if len(eval_slice) > 0 and "Model" in eval_slice.columns:
                for m, g in eval_slice.groupby("Model", dropna=False):
                    g2 = g.reset_index(drop=True).copy()
                    g2["_ord"] = range(len(g2))
                    tmpl_rows_by_model[safe_str_optional(m)] = g2

            # (1) Template-instance observations: ONE per (template row) with response in observation.description
            for model_name, tmpl_df in tmpl_rows_by_model.items():
                resp_m = resp_by_model.get(model_name)
                max_n = len(tmpl_df) if tmpl_df is not None else 0
                for i in range(max_n):
                    trow = tmpl_df.iloc[i]
                    template_text = safe_str_optional(trow.get("Template", "")).strip() or NA_STR
                    ds_id = ensure_template_dataset(project_id, project_name, metric_category, language, template_text)
                    tkey = template_key(template_text)

                    # align response by row order
                    response_text = ""
                    instance_val = ""
                    if resp_m is not None and i < len(resp_m):
                        response_text = safe_str_optional(resp_m.iloc[i].get("Response", ""))
                        instance_val = safe_str_optional(resp_m.iloc[i].get("Instance", ""))

                    # Observation per template-instance response
                    obs_id = stable_int(
                        f"observation::template::{project_name}/{metric_category}::{language}::{ts}::{model_name}::{tkey}::ord{i}"
                    )
                    desc = (response_text or "").strip()
                    if len(desc) > OBS_DESC_MAXLEN:
                        desc = desc[:OBS_DESC_MAXLEN]

                    rows["observation"].append({
                        "id": obs_id,
                        "observer": safe_str_required(DEFAULT_OBSERVER)[:100],
                        "whenObserved": when_obs,
                        "tool_id": tool_id,
                        "dataset_id": ds_id,
                        "eval_id": eval_id,
                        "name": safe_str_required(f"{metric_category}:{model_name}:template[{tkey}]")[:100],
                        "description": safe_str_required(desc),
                    })

                    # Measurand element for this template-instance:
                    # - element.name MUST reference dataset id for the template
                    # - element.description MUST contain populated template instance (filled prompt)
                    meas_eid = ensure_element(
                        "element",
                        str(ds_id),  # dataset id reference
                        (instance_val or NA_STR),
                        project_id
                    )

                    # Link measurand element to evaluation (bidirectional retrieval)
                    rows["evaluation_element"].append({"ref": meas_eid, "eval": eval_id})
                    rows["evaluates_eval"].append({"evaluates": meas_eid, "evalu": eval_id})

                    def add_measure(metric_name: str, measurand_eid: int, value: Any) -> None:
                        mid = metric_ids[metric_name]
                        meas_id = stable_int(f"measure::{obs_id}::{mid}::{measurand_eid}")
                        v = safe_str_required(value)
                        if len(v) > MEASURE_VALUE_MAXLEN:
                            v = v[:MEASURE_VALUE_MAXLEN]
                        rows["measure"].append({
                            "id": meas_id,
                            "value": v,
                            "error": safe_str_required(MEASURE_ERROR_DEFAULT)[:100],
                            "uncertainty": float(MEASURE_UNCERTAINTY_DEFAULT),
                            "unit": safe_str_required(MEASURE_UNIT_DEFAULT)[:100],
                            "measurand_id": measurand_eid,
                            "metric_id": mid,
                            "observation_id": obs_id
                        })

                    # Attach per-row metrics (no response_text measure)
                    add_measure("evaluation", meas_eid, trow.get("Evaluation"))
                    add_measure("oracle_prediction", meas_eid, trow.get("Oracle Prediction"))
                    add_measure("oracle_evaluation", meas_eid, trow.get("Oracle Evaluation"))

            # (2) Run-level observation for globals (optional)
            run_ctx_ds_id = ensure_template_dataset(project_id, project_name, metric_category, language, "__RUN_CONTEXT__")
            run_obs_id = stable_int(f"observation::run::{project_name}/{metric_category}::{language}::{ts}::{cfg_sig}")

            rows["observation"].append({
                "id": run_obs_id,
                "observer": safe_str_required(DEFAULT_OBSERVER)[:100],
                "whenObserved": when_obs,
                "tool_id": tool_id,
                "dataset_id": run_ctx_ds_id,
                "eval_id": eval_id,
                "name": safe_str_required(f"{metric_category}:RUN")[:100],
                "description": safe_str_required(f"Run-level global summary for {project_name}/{metric_category} at {ts} ({language})"),
            })

            if len(global_slice) > 0:
                for j, grow in global_slice.reset_index(drop=True).iterrows():
                    # Separate measurand for each global row
                    g_eid = ensure_element(
                        "element",
                        str(run_ctx_ds_id),
                        f"Global summary row {j} | project={project_name} concern={metric_category} lang={language} ts={ts}",
                        project_id
                    )
                    # Link to evaluation too
                    rows["evaluation_element"].append({"ref": g_eid, "eval": eval_id})
                    rows["evaluates_eval"].append({"evaluates": g_eid, "evalu": eval_id})

                    for col in ["Passed Nr", "Failed Nr", "Error Nr", "Passed Pct", "Failed Pct", "Total", "Tolerance", "Tolerance Evaluation"]:
                        if col in global_slice.columns:
                            mid = metric_ids[col]
                            meas_id = stable_int(f"measure::{run_obs_id}::{mid}::{g_eid}")
                            v = safe_str_required(grow.get(col))
                            if len(v) > MEASURE_VALUE_MAXLEN:
                                v = v[:MEASURE_VALUE_MAXLEN]
                            rows["measure"].append({
                                "id": meas_id,
                                "value": v,
                                "error": safe_str_required(MEASURE_ERROR_DEFAULT)[:100],
                                "uncertainty": float(MEASURE_UNCERTAINTY_DEFAULT),
                                "unit": safe_str_required(MEASURE_UNIT_DEFAULT)[:100],
                                "measurand_id": g_eid,
                                "metric_id": mid,
                                "observation_id": run_obs_id
                            })

    # Write out CSVs
    table_keys = {
        "project": ["id"],
        "tool": ["id"],
        "configuration": ["id"],
        "confparam": ["id"],
        "datashape": ["id"],
        "element": ["id"],
        "dataset": ["id"],
        "model": ["id"],
        "metriccategory": ["id"],
        "metriccategory_metric": ["id"],
        "evaluation": ["id"],
        "evaluation_element": ["ref", "eval"],
        "evaluates_eval": ["evaluates", "evalu"],
        "metric": ["id"],
        "direct": ["id"],
        "derived": ["id"],
        "observation": ["id"],
        "measure": ["id"],
    }

    for table, buf in rows.items():
        if not buf:
            continue
        df = pd.DataFrame(buf)
        out_csv = out_dir / f"{table}.csv"
        upsert_append(out_csv, df, table_keys[table])

    print(f"✅ Accumulated CSVs written to: {out_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Repo root")
    ap.add_argument("--manifest", default="manifest.json", help="Path to manifest.json (relative to repo)")
    ap.add_argument("--out", default="data/data_accumulated", help="Output directory")
    args = ap.parse_args()

    main(
        repo_root=Path(args.repo).resolve(),
        manifest_path=Path(args.manifest),
        out_dir=Path(args.repo).resolve() / Path(args.out),
    )
