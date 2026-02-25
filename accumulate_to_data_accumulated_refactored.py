#!/usr/bin/env python3
from __future__ import annotations

"""
Accumulator (refactored Feb 2026)

Key refactor goals (based on colleague feedback):
- Treat top-level directory names under data/ as *Projects* (e.g., experiments, mistral, microcreditagent).
- Treat "ageism", "xenophobia", "religion", ... as *MetricCategory* (derived from test_name like test-ageism).
- Evaluation = stable configuration grouping (same project + metric_category + language + model-set + config_signature).
- Observation = the datetime folder (timestamp_dir) i.e., a single run execution. Measures hang off that observation.
- Ensure relationship traversals work from BOTH sides by writing:
  - evaluation_element (evaluation -> element)
  - evaluates_eval   (element -> evaluation)  [mirror edges]
- Populate MetricCategory (+ optional MetricCategory↔Metric mapping) so category-level queries are possible.
"""

import json
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ----------------------------
# Configuration / constants
# ----------------------------

DEFAULT_OBSERVER = "MLABiTe"

TOOL_NAME = "MLABiTe"
TOOL_SOURCE = "MLABiTe"
TOOL_LICENSING = "Open_Source"          # enum string in your DB

PROJECT_STATUS = "Pending"              # enum string in your DB
EVALUATION_STATUS = "Done"              # enum string in your DB

DATASET_LICENSING_DEFAULT = "Proprietary"
DATASET_TYPE_DEFAULT = "Test"           # enum string in your DB

MODEL_DATA_DEFAULT = "NA"

MEASURE_ERROR_DEFAULT = "NA"
MEASURE_UNIT_DEFAULT = "NA"
MEASURE_UNCERTAINTY_DEFAULT = 0.0

# You requested to increase this to 10k (also update sql_alchemy.py accordingly)
MEASURE_VALUE_MAXLEN = 10000

# If the manifest doesn't provide a provider_family (project), use this:
DEFAULT_PROJECT_NAME = "default_project"

# ----------------------------
# Utilities
# ----------------------------

def stable_int(key: str) -> int:
    """
    Deterministic int ID from a key.
    Fits into signed 32-bit range to stay SQLite-friendly.
    """
    h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    n = int(h[:8], 16)  # 32-bit
    # avoid 0
    return (n % 2_000_000_000) + 1


def parse_timestamp_dir(ts: str) -> str:
    """
    Convert YYYYMMDD_HHMMSS -> ISO datetime string for whenObserved.
    """
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.isoformat(sep=" ")
    except Exception:
        return datetime.utcnow().isoformat(sep=" ")


def sniff_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    return ";" if sample.count(";") > sample.count(",") else ","


def read_csv_flex(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
    - sniff delimiter
    - try fast C engine
    - fallback to Python engine with relaxed parsing
    """
    sep = sniff_sep(path)

    # 1) Fast path
    try:
        return pd.read_csv(path, sep=sep, engine="c")
    except Exception:
        pass

    # 2) Python engine fallback (more tolerant)
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
        # 3) last resort: try reading with the opposite delimiter
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
    Responses often contain commas/semicolons/newlines that break normal CSV parsing.
    We parse line-by-line and treat everything after the third separator as the Response text.

    Expected columns (minimum): Provider, Model, Instance, Response
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

        rows.append({
            "Provider": provider,
            "Model": model,
            "Instance": instance,
            "Response": response
        })

    return pd.DataFrame(rows, columns=["Provider", "Model", "Instance", "Response"])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def upsert_append(out_csv: Path, df: pd.DataFrame, key_cols: List[str]) -> None:
    """
    Append + dedupe by key cols.
    """
    if out_csv.exists():
        prev = read_csv_flex(out_csv)
        merged = pd.concat([prev, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="first")
        merged.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False)


def norm_path(p: str) -> Path:
    return Path(p.replace("\\", os.sep))


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def normalize_metric_category(test_name: str) -> str:
    """
    Map 'test-ageism' -> 'ageism'
    If not matching, return as-is.
    """
    t = (test_name or "").strip()
    if t.lower().startswith("test-"):
        return t[5:]
    return t or "unknown_category"


def canonical_json(obj: Any) -> str:
    """
    Canonical JSON string for stable hashing (sort keys, no whitespace noise).
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_signature(cfg: dict) -> str:
    """
    Short stable signature for configuration grouping.
    """
    return hashlib.sha1(canonical_json(cfg).encode("utf-8", errors="ignore")).hexdigest()[:16]


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

def main(
    repo_root: Path,
    manifest_path: Path,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)

    manifest = json.loads((repo_root / "data" / "data_accumulated" / manifest_path).read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])

    # ---- Accumulated row buffers (list of dicts per table)
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
        "observation",
        "measure",
    ]}

    # ---- Shared baseline datashape
    datashape_id = stable_int("datashape::generic")
    rows["datashape"].append({
        "id": datashape_id,
        "accepted_target_values": "NA"
    })

    # ---- Global tool (single row; observations point to it)
    tool_id = stable_int(f"tool::{TOOL_NAME}")
    rows["tool"].append({
        "id": tool_id,
        "source": TOOL_SOURCE,
        "version": "NA",
        "name": TOOL_NAME,
        "licensing": TOOL_LICENSING
    })

    # ---- Shared "ModelRegistry" dataset (for model.dataset_id FK)
    model_registry_dataset_element_id = stable_int("element::dataset::ModelRegistry")
    rows["element"].append({
        "id": model_registry_dataset_element_id,
        "project_id": None,
        "type_spec": "dataset",
        "name": "ModelRegistry",
        "description": "Dataset placeholder for model registry linkage"
    })
    rows["dataset"].append({
        "id": model_registry_dataset_element_id,
        "source": "ModelRegistry",
        "version": "NA",
        "licensing": "Open_Source",
        "dataset_type": "Test",
        "datashape_id": datashape_id,
    })

    # ---- Metrics: created dynamically from encountered CSVs
    metric_ids: Dict[str, int] = {}

    def ensure_metric(metric_name: str, description: str = "") -> int:
        key = metric_name.strip()
        if key in metric_ids:
            return metric_ids[key]
        mid = stable_int(f"metric::{key}")
        metric_ids[key] = mid
        rows["metric"].append({
            "id": mid,
            "type_spec": "Direct",
            "name": key,
            "description": description or key
        })
        rows["direct"].append({"id": mid})
        return mid

    # ---- Elements
    def ensure_element(type_spec: str, name: str, description: str, project_id: Optional[int]) -> int:
        eid = stable_int(f"element::{type_spec}::{name}")
        rows["element"].append({
            "id": eid,
            "project_id": project_id,
            "type_spec": type_spec,
            "name": name,
            "description": description
        })
        return eid

    # ---- Datasets (per project+category, plus per Template)
    def ensure_dataset(project_id: int, project_name: str, metric_category: str, template_name: str) -> int:
        """
        Create element + dataset rows. Dataset id == element id (polymorphic).
        """
        ds_elem_id = stable_int(f"element::dataset::{project_name}::{metric_category}::{template_name}")
        rows["element"].append({
            "id": ds_elem_id,
            "project_id": project_id,
            "type_spec": "dataset",
            "name": template_name,
            "description": f"Dataset (template={template_name}) for {project_name}/{metric_category}"
        })
        rows["dataset"].append({
            "id": ds_elem_id,
            "source": f"{project_name}/{metric_category}",
            "version": "NA",
            "licensing": DATASET_LICENSING_DEFAULT,
            "dataset_type": DATASET_TYPE_DEFAULT,
            "datashape_id": datashape_id,
        })
        return ds_elem_id

    # ---- Models
    def ensure_model(pid: str, project_id: int) -> int:
        """
        Create element + model rows. Model id == element id.
        """
        info = lookup_model(pid)
        model_elem_id = stable_int(f"element::model::{pid}")
        rows["element"].append({
            "id": model_elem_id,
            "project_id": project_id,
            "type_spec": "model",
            "name": pid,
            "description": f"{info.get('source','Unknown')} | {info.get('canonical',pid)}"
        })
        rows["model"].append({
            "id": model_elem_id,
            "pid": pid,
            "data": info.get("canonical", pid) or MODEL_DATA_DEFAULT,
            "source": info.get("source", "Unknown"),
            "licensing": info.get("licensing", "Proprietary"),
            "dataset_id": model_registry_dataset_element_id,
        })
        return model_elem_id

    # ---- MetricCategory
    metric_category_ids: Dict[str, int] = {}

    def ensure_metric_category(name: str) -> int:
        n = name.strip() or "unknown_category"
        if n in metric_category_ids:
            return metric_category_ids[n]
        mc_id = stable_int(f"metriccategory::{n}")
        metric_category_ids[n] = mc_id
        rows["metriccategory"].append({
            "id": mc_id,
            "name": n,
            "description": f"Metric category '{n}' derived from test_name"
        })
        return mc_id

    # ---- Evaluation registry to group multiple timestamp runs into one evaluation
    eval_registry: Dict[Tuple[str, str, str, str, str], int] = {}  # (project, category, language, models_key, cfg_sig) -> eval_id
    config_registry: Dict[str, int] = {}  # cfg_sig -> config_id

    def get_or_create_configuration(project_name: str, metric_category: str, cfg: dict) -> Tuple[int, str]:
        sig = config_signature(cfg)
        if sig in config_registry:
            return config_registry[sig], sig

        config_id = stable_int(f"configuration::{project_name}::{metric_category}::{sig}")
        config_registry[sig] = config_id
        rows["configuration"].append({
            "id": config_id,
            "name": f"config_{project_name}_{metric_category}_{sig}",
            "description": f"Config for {project_name}/{metric_category} (sig={sig})"
        })

        # Flatten top-level keys into confparam rows
        for k, v in cfg.items():
            cp_id = stable_int(f"confparam::{sig}::{k}")
            rows["confparam"].append({
                "id": cp_id,
                "param_type": "json",
                "value": json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v,
                "conf_id": config_id,
                "name": k,
                "description": "from config.json"
            })

        return config_id, sig

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

        rows["evaluation"].append({
            "id": eval_id,
            "status": EVALUATION_STATUS,
            "config_id": config_id,
            "project_id": project_id
        })

        # Dimension / linkage elements (written in BOTH association tables)
        def link_eval_element(eid: int) -> None:
            rows["evaluation_element"].append({"ref": eid, "eval": eval_id})
            rows["evaluates_eval"].append({"evaluates": eid, "evalu": eval_id})

        # language element
        lang_eid = ensure_element("element", f"Language={language}", "Language dimension", project_id)
        link_eval_element(lang_eid)

        # metric category as element (so you can traverse through evaluation_element)
        cat_eid = ensure_element("element", f"MetricCategory={metric_category}", "Metric category dimension", project_id)
        link_eval_element(cat_eid)

        # models
        for pid in sorted(set([p for p in model_pids if p.strip()])):
            mid = ensure_model(pid, project_id)
            link_eval_element(mid)

        return eval_id

    # ---- Ensure baseline metrics (these are used across categories)
    ensure_metric("evaluation", "Row-level evaluation result")
    ensure_metric("oracle_prediction", "Row-level oracle prediction")
    ensure_metric("oracle_evaluation", "Row-level oracle evaluation")
    ensure_metric("response_text", "Raw model response text")
    for m in ["Passed Nr", "Failed Nr", "Error Nr", "Passed Pct", "Failed Pct", "Total", "Tolerance", "Tolerance Evaluation"]:
        ensure_metric(m, f"Global metric {m}")

    # ----------------------------
    # Iterate runs
    # ----------------------------
    for run in runs:
        ts = run["timestamp_dir"]  # observation identity
        test_name = run.get("test_name") or "unknown_test"
        metric_category = normalize_metric_category(test_name)
        language = run.get("language") or "unknown_lang"
        paths = run["paths"]

        project_name = run.get("provider_family") or DEFAULT_PROJECT_NAME

        # ---- project (one per project_name)
        project_id = stable_int(f"project::{project_name}")
        rows["project"].append({
            "id": project_id,
            "name": project_name,
            "status": PROJECT_STATUS
        })

        # ---- metric category
        mc_id = ensure_metric_category(metric_category)

        # ---- load config.json
        cfg_path = repo_root / norm_path(paths["config_json"])
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        # ---- determine models (from config.json aiModels[], fallback to CSV)
        evals_df = read_csv_flex(repo_root / norm_path(paths["evals_csv"]))
        global_df = read_csv_flex(repo_root / norm_path(paths["global_csv"]))

        resp_path = repo_root / norm_path(paths["responses_csv"])
        try:
            resp_df = read_csv_flex(resp_path)
        except Exception:
            resp_df = read_responses_csv_robust(resp_path)
        if "Response" not in resp_df.columns and "Instance" not in resp_df.columns:
            resp_df = read_responses_csv_robust(resp_path)

        model_pids = cfg.get("aiModels") or []
        if not model_pids:
            if "Model" in evals_df.columns and len(evals_df) > 0:
                model_pids = [safe_str(evals_df["Model"].iloc[0])]
            else:
                model_pids = ["UnknownModel"]

        # ---- config grouping
        config_id, cfg_sig = get_or_create_configuration(project_name, metric_category, cfg)

        # ---- evaluation grouping
        eval_id = get_or_create_evaluation(
            project_id=project_id,
            project_name=project_name,
            metric_category=metric_category,
            language=language,
            model_pids=model_pids,
            config_id=config_id,
            cfg_sig=cfg_sig
        )

        # ---- Link MetricCategory -> Metric (optional, but useful for category browsing)
        # We link the known core metrics to every category we encounter.
        for metric_name, mid in metric_ids.items():
            mm_id = stable_int(f"metriccategory_metric::{mc_id}::{mid}")
            rows["metriccategory_metric"].append({
                "id": mm_id,
                "metriccategory_id": mc_id,
                "metric_id": mid
            })

        # ---- Datasets for this project/category
        # We'll always create __NO_TEMPLATE__ and __RUN_CONTEXT__.
        ds_no_template = ensure_dataset(project_id, project_name, metric_category, "__NO_TEMPLATE__")
        ds_run_context = ensure_dataset(project_id, project_name, metric_category, "__RUN_CONTEXT__")

        # Also create template-specific datasets (and link as evaluation elements)
        templates: List[str] = []
        if "Template" in evals_df.columns:
            templates = [t for t in evals_df["Template"].dropna().unique().tolist()]
        if not templates:
            templates = []

        dataset_by_template: Dict[str, int] = {"__NO_TEMPLATE__": ds_no_template}
        for t in templates:
            tt = safe_str(t).strip() or "__NO_TEMPLATE__"
            dataset_by_template[tt] = ensure_dataset(project_id, project_name, metric_category, tt)

            # Link template as element to evaluation (and mirror)
            tmpl_eid = ensure_element("element", f"Template={tt}", "Template dimension", project_id)
            rows["evaluation_element"].append({"ref": tmpl_eid, "eval": eval_id})
            rows["evaluates_eval"].append({"evaluates": tmpl_eid, "evalu": eval_id})

        # ---- Observation = datetime folder (timestamp_dir)
        when_obs = parse_timestamp_dir(ts)
        obs_id = stable_int(f"observation::run::{project_name}::{metric_category}::{cfg_sig}::{ts}::{language}")
        rows["observation"].append({
            "id": obs_id,
            "observer": DEFAULT_OBSERVER,
            "whenObserved": when_obs,
            "tool_id": tool_id,
            "dataset_id": ds_run_context,  # placeholder; measure measurands disambiguate slices
            "eval_id": eval_id,
            "name": f"{metric_category}",
            "description": f"Run observation for {project_name}/{metric_category} at {ts} ({language})"
        })

        # ---- Measures
        # We store all row-level/global/response outputs as measures attached to the run observation,
        # each with its own measurand element describing the slice/instance.

        def add_measure(metric_name: str, measurand_eid: int, value: Any) -> None:
            mid = metric_ids[metric_name]
            meas_id = stable_int(f"measure::{obs_id}::{mid}::{measurand_eid}")
            rows["measure"].append({
                "id": meas_id,
                "value": safe_str(value)[:MEASURE_VALUE_MAXLEN],
                "error": MEASURE_ERROR_DEFAULT,
                "uncertainty": MEASURE_UNCERTAINTY_DEFAULT,
                "unit": MEASURE_UNIT_DEFAULT,
                "measurand_id": measurand_eid,
                "metric_id": mid,
                "observation_id": obs_id
            })

        # (A) evaluations.csv rows
        for idx, r in evals_df.iterrows():
            concern = safe_str(r.get("Concern", "")).strip()
            input_type = safe_str(r.get("Input Type", "")).strip()
            reflection_type = safe_str(r.get("Reflection Type", "")).strip()
            raw_template = r.get("Template", None)
            template = safe_str(raw_template).strip() or "__NO_TEMPLATE__"

            # Create a measurand element that uniquely identifies the slice row
            slice_key = f"{project_name}|{metric_category}|{cfg_sig}|{ts}|{language}|{template}|{concern}|{input_type}|{reflection_type}|row={idx}"
            meas_name = f"EvalRow={hashlib.sha1(slice_key.encode()).hexdigest()[:12]}"
            meas_desc = f"{template} | {concern} | {input_type} | {reflection_type} | row {idx}"
            meas_eid = ensure_element("element", meas_name, meas_desc, project_id)

            # Also record dimensions as elements linked to evaluation (both directions)
            for dim_name, dim_val, dim_desc in [
                ("Concern", concern, "Concern dimension"),
                ("InputType", input_type, "Input Type dimension"),
                ("ReflectionType", reflection_type, "Reflection Type dimension"),
            ]:
                if dim_val:
                    deid = ensure_element("element", f"{dim_name}={dim_val}", dim_desc, project_id)
                    rows["evaluation_element"].append({"ref": deid, "eval": eval_id})
                    rows["evaluates_eval"].append({"evaluates": deid, "evalu": eval_id})

            add_measure("evaluation", meas_eid, r.get("Evaluation"))
            add_measure("oracle_prediction", meas_eid, r.get("Oracle Prediction"))
            add_measure("oracle_evaluation", meas_eid, r.get("Oracle Evaluation"))

        # (B) global_evaluation.csv rows
        for idx, r in global_df.iterrows():
            concern = safe_str(r.get("Concern", "")).strip()
            input_type = safe_str(r.get("Input Type", "")).strip()
            reflection_type = safe_str(r.get("Reflection Type", "")).strip()

            slice_key = f"{project_name}|{metric_category}|{cfg_sig}|{ts}|{language}|GLOBAL|{concern}|{input_type}|{reflection_type}|row={idx}"
            meas_name = f"GlobalRow={hashlib.sha1(slice_key.encode()).hexdigest()[:12]}"
            meas_desc = f"GLOBAL | {concern} | {input_type} | {reflection_type} | row {idx}"
            meas_eid = ensure_element("element", meas_name, meas_desc, project_id)

            for col in ["Passed Nr", "Failed Nr", "Error Nr", "Passed Pct", "Failed Pct", "Total", "Tolerance", "Tolerance Evaluation"]:
                if col in global_df.columns:
                    add_measure(col, meas_eid, r.get(col))

        # (C) responses.csv rows
        if "Instance" in resp_df.columns:
            for idx, r in resp_df.iterrows():
                instance = safe_str(r.get("Instance", f"row{idx}")).strip() or f"row{idx}"
                meas_eid = ensure_element("element", f"Instance={instance}", "Instance dimension", project_id)

                # Link instance as a dimension of the evaluation (both directions)
                rows["evaluation_element"].append({"ref": meas_eid, "eval": eval_id})
                rows["evaluates_eval"].append({"evaluates": meas_eid, "evalu": eval_id})

                add_measure("response_text", meas_eid, r.get("Response", ""))

    # ---- Write tables to data_accumulated (upsert style)
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
