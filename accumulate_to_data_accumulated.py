#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

# You asked to increase this to 10k (also update sql_alchemy.py accordingly)
MEASURE_VALUE_MAXLEN = 10000


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
    # crude but effective
    return ";" if sample.count(";") > sample.count(",") else ","


# def read_csv_flex(path: Path) -> pd.DataFrame:
#     sep = sniff_sep(path)
#     return pd.read_csv(path, sep=sep)
# PSA

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
            on_bad_lines="skip",   # pandas >= 1.3
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
    If Provider/Model are missing, we still keep Instance/Response.
    """
    sep = sniff_sep(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return pd.DataFrame(columns=["Provider", "Model", "Instance", "Response"])

    header = lines[0].split(sep)
    # Normalize header guesses
    # We will always output these canonical cols:
    out_cols = ["Provider", "Model", "Instance", "Response"]

    rows = []
    for raw in lines[1:]:
        if not raw.strip():
            continue

        parts = raw.split(sep)
        if len(parts) < 2:
            continue

        # If it looks like it already has proper 4+ columns, try to join extras to Response
        if len(parts) >= 4:
            provider = parts[0].strip()
            model = parts[1].strip()
            instance = parts[2].strip()
            response = sep.join(parts[3:]).strip()
        else:
            # Minimal fallback: last token is instance, rest is response
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

    return pd.DataFrame(rows, columns=out_cols)


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


# ----------------------------
# Model registry (starter)
# Extend this as you like.
# ----------------------------

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # OpenAI / Azure OpenAI naming variants
    "OpenAIGPT35Turbo": {"canonical": "gpt-35-turbo", "source": "OpenAI", "licensing": "Proprietary"},
    "OpenAIGPT4o": {"canonical": "gpt-4o", "source": "OpenAI", "licensing": "Proprietary"},
    "OpenAIGPT4oMini": {"canonical": "gpt-4o-mini", "source": "OpenAI", "licensing": "Proprietary"},

    # Mistral
    "MistralMedium": {"canonical": "mistral-medium-2505", "source": "Mistral", "licensing": "Proprietary"},
    "MistralLarge": {"canonical": "mistral-large", "source": "Mistral", "licensing": "Proprietary"},

    # Anthropic
    "ClaudeSonnet": {"canonical": "claude-sonnet", "source": "Anthropic", "licensing": "Proprietary"},

    # Google
    "GeminiPro": {"canonical": "gemini-pro", "source": "Google", "licensing": "Proprietary"},

    # Meta (some are open weights)
    "Llama3": {"canonical": "llama-3", "source": "Meta", "licensing": "Open_Source"},

    # DeepSeek
    "DeepSeekR1": {"canonical": "deepseek-r1", "source": "DeepSeek", "licensing": "Open_Source"},

    # Microsoft
    "Phi": {"canonical": "phi", "source": "Microsoft", "licensing": "Open_Source"},

    # xAI
    "Grok": {"canonical": "grok", "source": "xAI", "licensing": "Proprietary"},
}


def lookup_model(pid: str) -> Dict[str, str]:
    # exact match first
    if pid in MODEL_REGISTRY:
        return MODEL_REGISTRY[pid]

    # fuzzy match: strip common noise
    compact = re.sub(r"[^a-zA-Z0-9]+", "", pid).lower()
    for k, v in MODEL_REGISTRY.items():
        if re.sub(r"[^a-zA-Z0-9]+", "", k).lower() == compact:
            return v

    # fallback heuristic
    src = "Unknown"
    lic = "Proprietary"
    if "llama" in compact or "phi" in compact or "deepseek" in compact:
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
        "project", "tool", "configuration", "confparam",
        "datashape", "element", "dataset", "model",
        "evaluation", "evaluation_element",
        "metric", "direct",
        "observation", "measure",
    ]}

    # ---- Shared baseline datashape
    datashape_id = stable_int("datashape::generic")
    rows["datashape"].append({
        "id": datashape_id,
        "accepted_target_values": "NA"
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

    # ---- Define metrics we’ll create dynamically (names discovered from CSVs)
    metric_ids: Dict[str, int] = {}

    def ensure_metric(metric_name: str, description: str = "") -> int:
        if metric_name in metric_ids:
            return metric_ids[metric_name]
        mid = stable_int(f"metric::{metric_name}")
        metric_ids[metric_name] = mid
        rows["metric"].append({
            "id": mid,
            "type_spec": "Direct",
            "name": metric_name,
            "description": description or metric_name
        })
        rows["direct"].append({"id": mid})
        return mid

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

    def ensure_dataset(template_name: str, project_id: int, source: str) -> int:
        """
        Create element + dataset rows. Dataset id == element id (polymorphic).
        """
        ds_elem_id = stable_int(f"element::dataset::{source}::{template_name}")
        rows["element"].append({
            "id": ds_elem_id,
            "project_id": project_id,
            "type_spec": "dataset",
            "name": template_name,
            "description": f"Dataset derived from Template for {source}"
        })
        rows["dataset"].append({
            "id": ds_elem_id,
            "source": source,
            "version": "NA",
            "licensing": DATASET_LICENSING_DEFAULT,
            "dataset_type": DATASET_TYPE_DEFAULT,
            "datashape_id": datashape_id,
        })
        return ds_elem_id

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

    for run in runs:
        ts = run["timestamp_dir"]                     # UNIQUE run identity
        test_name = run.get("test_name") or "unknown_test"
        language = run.get("language") or "unknown_lang"
        paths = run["paths"]

        # ---- project (one per test_name)
        project_id = stable_int(f"project::{test_name}")
        rows["project"].append({
            "id": project_id,
            "name": test_name,
            "status": PROJECT_STATUS
        })

        # ---- tool (per timestamp_dir as requested)
        tool_id = stable_int(f"tool::{TOOL_NAME}::{ts}")
        rows["tool"].append({
            "id": tool_id,
            "source": TOOL_SOURCE,
            "version": ts,
            "name": TOOL_NAME,
            "licensing": TOOL_LICENSING
        })

        # ---- configuration (per timestamp_dir)
        config_id = stable_int(f"config::{ts}")
        rows["configuration"].append({
            "id": config_id,
            "name": f"config_{ts}",
            "description": f"Config for {test_name} {ts}"
        })

        cfg_path = repo_root / norm_path(paths["config_json"])
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        # confparam rows: flatten only top-level keys (stringify values)
        for k, v in cfg.items():
            cp_id = stable_int(f"confparam::{ts}::{k}")
            rows["confparam"].append({
                "id": cp_id,
                "param_type": "json",
                "value": json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v,
                "conf_id": config_id,
                "name": k,
                "description": "from config.json"
            })

        # ---- evaluation (ONE per timestamp_dir per your rule)
        eval_id = stable_int(f"evaluation::{ts}")
        rows["evaluation"].append({
            "id": eval_id,
            "status": EVALUATION_STATUS,
            "config_id": config_id,
            "project_id": project_id
        })

        # ---- dimension elements attached to evaluation (language, test_name)
        lang_eid = ensure_element("element", f"Language={language}", "Language dimension", project_id)
        rows["evaluation_element"].append({"ref": lang_eid, "eval": eval_id})

        test_eid = ensure_element("element", f"Test={test_name}", "Test name dimension", project_id)
        rows["evaluation_element"].append({"ref": test_eid, "eval": eval_id})

        # ---- load run CSVs
        evals_df = read_csv_flex(repo_root / norm_path(paths["evals_csv"]))
        global_df = read_csv_flex(repo_root / norm_path(paths["global_csv"]))
        # resp_df = read_csv_flex(repo_root / norm_path(paths["responses_csv"]))
        resp_path = repo_root / norm_path(paths["responses_csv"])
        try:
            resp_df = read_csv_flex(resp_path)
        except Exception:
            resp_df = read_responses_csv_robust(resp_path)

        # If parsing succeeded but columns are wrong/missing, fallback anyway
        if "Response" not in resp_df.columns and "Instance" not in resp_df.columns:
            resp_df = read_responses_csv_robust(resp_path)


        # ---- aiModels from config.json
        ai_models = cfg.get("aiModels") or []
        if not ai_models:
            # fallback: attempt from evals CSV
            if "Model" in evals_df.columns and len(evals_df) > 0:
                ai_models = [safe_str(evals_df["Model"].iloc[0])]
            else:
                ai_models = ["UnknownModel"]

        # Attach model elements to evaluation
        for pid in ai_models:
            model_elem_id = ensure_model(pid, project_id)
            rows["evaluation_element"].append({"ref": model_elem_id, "eval": eval_id})

        # ---- dataset elements derived from Template column
        templates: List[str] = []
        if "Template" in evals_df.columns:
            templates = [t for t in evals_df["Template"].dropna().unique().tolist()]
        if not templates:
            templates = ["__NO_TEMPLATE__"]

        dataset_ids_by_template: Dict[str, int] = {}
        for t in templates:
            # Always ensure a fallback dataset exists
            if "__NO_TEMPLATE__" not in dataset_ids_by_template:
                ds_id = ensure_dataset(template_name="__NO_TEMPLATE__", project_id=project_id, source=test_name)
                dataset_ids_by_template["__NO_TEMPLATE__"] = ds_id
            ds_id = ensure_dataset(template_name=str(t), project_id=project_id, source=test_name)
            dataset_ids_by_template[str(t)] = ds_id
            # attach template element to evaluation too
            tmpl_eid = ensure_element("element", f"Template={t}", "Template dimension", project_id)
            rows["evaluation_element"].append({"ref": tmpl_eid, "eval": eval_id})

        # ---- metrics we will use (ensure now)
        # evaluations.csv row-level
        ensure_metric("evaluation", "Row-level evaluation result")
        ensure_metric("oracle_prediction", "Row-level oracle prediction")
        ensure_metric("oracle_evaluation", "Row-level oracle evaluation")
        # global_evaluation.csv
        for m in ["Passed Nr", "Failed Nr", "Error Nr", "Passed Pct", "Failed Pct", "Total", "Tolerance", "Tolerance Evaluation"]:
            ensure_metric(m, f"Global metric {m}")
        # responses.csv
        ensure_metric("response_text", "Raw model response text")

        # ---- observations + measures
        when_obs = parse_timestamp_dir(ts)

        # (A) evaluations.csv -> observation per row (dataset from Template)
        for idx, row in evals_df.iterrows():
            concern = safe_str(row.get("Concern", ""))
            input_type = safe_str(row.get("Input Type", ""))
            reflection_type = safe_str(row.get("Reflection Type", ""))
            # template = safe_str(row.get("Template", "__NO_TEMPLATE__"))
            raw_template = row.get("Template", None)
            template = safe_str(raw_template).strip()
            if not template:
                template = "__NO_TEMPLATE__"

            # dimension elements (attached to evaluation already, but we also create per value to reference as measurand)
            c_eid = ensure_element("element", f"Concern={concern}", "Concern dimension", project_id)
            it_eid = ensure_element("element", f"InputType={input_type}", "Input Type dimension", project_id)
            rt_eid = ensure_element("element", f"ReflectionType={reflection_type}", "Reflection Type dimension", project_id)

            rows["evaluation_element"].append({"ref": c_eid, "eval": eval_id})
            rows["evaluation_element"].append({"ref": it_eid, "eval": eval_id})
            rows["evaluation_element"].append({"ref": rt_eid, "eval": eval_id})

            # measurand element = slice + template + row index (stable)
            slice_key = f"{ts}|{language}|{template}|{concern}|{input_type}|{reflection_type}|row={idx}"
            meas_eid = ensure_element("element", f"Measurand={hashlib.sha1(slice_key.encode()).hexdigest()[:12]}",
                                      f"{template} | {concern} | {input_type} | {reflection_type} | row {idx}", project_id)

            # dataset_id = dataset_ids_by_template.get(template) or dataset_ids_by_template["__NO_TEMPLATE__"]
            dataset_id = dataset_ids_by_template.get(template, dataset_ids_by_template["__NO_TEMPLATE__"])


            obs_id = stable_int(f"observation::{eval_id}::{template}::{idx}")
            rows["observation"].append({
                "id": obs_id,
                "observer": DEFAULT_OBSERVER,
                "whenObserved": when_obs,
                "tool_id": tool_id,
                "dataset_id": dataset_id,
                "eval_id": eval_id,
                "name": concern or "evaluation_row",
                "description": f"Row-level evaluation | template={template}"
            })

            # measures for this observation (3 metrics)
            def add_measure(metric_name: str, value: Any) -> None:
                mid = metric_ids[metric_name]
                meas_id = stable_int(f"measure::{obs_id}::{mid}::{meas_eid}")
                rows["measure"].append({
                    "id": meas_id,
                    "value": safe_str(value)[:MEASURE_VALUE_MAXLEN],
                    "error": MEASURE_ERROR_DEFAULT,
                    "uncertainty": MEASURE_UNCERTAINTY_DEFAULT,
                    "unit": MEASURE_UNIT_DEFAULT,
                    "measurand_id": meas_eid,
                    "metric_id": mid,
                    "observation_id": obs_id
                })

            add_measure("evaluation", row.get("Evaluation"))
            add_measure("oracle_prediction", row.get("Oracle Prediction"))
            add_measure("oracle_evaluation", row.get("Oracle Evaluation"))

        # (B) global_evaluation.csv -> observation per row
        for idx, row in global_df.iterrows():
            concern = safe_str(row.get("Concern", ""))
            input_type = safe_str(row.get("Input Type", ""))
            reflection_type = safe_str(row.get("Reflection Type", ""))

            slice_key = f"{ts}|{language}|GLOBAL|{concern}|{input_type}|{reflection_type}|row={idx}"
            meas_eid = ensure_element(
                "element",
                f"GlobalSlice={hashlib.sha1(slice_key.encode()).hexdigest()[:12]}",
                f"GLOBAL | {concern} | {input_type} | {reflection_type}",
                project_id
            )

            # dataset for global: re-use __NO_TEMPLATE__ dataset
            dataset_id = dataset_ids_by_template.get("__NO_TEMPLATE__") or list(dataset_ids_by_template.values())[0]

            obs_id = stable_int(f"observation::{eval_id}::global::{idx}")
            rows["observation"].append({
                "id": obs_id,
                "observer": DEFAULT_OBSERVER,
                "whenObserved": when_obs,
                "tool_id": tool_id,
                "dataset_id": dataset_id,
                "eval_id": eval_id,
                "name": concern or "global_evaluation",
                "description": "Global evaluation metrics"
            })

            for col in ["Passed Nr", "Failed Nr", "Error Nr", "Passed Pct", "Failed Pct", "Total", "Tolerance", "Tolerance Evaluation"]:
                if col in global_df.columns:
                    mid = metric_ids[col]
                    meas_id = stable_int(f"measure::{obs_id}::{mid}::{meas_eid}")
                    rows["measure"].append({
                        "id": meas_id,
                        "value": safe_str(row.get(col))[:MEASURE_VALUE_MAXLEN],
                        "error": MEASURE_ERROR_DEFAULT,
                        "uncertainty": MEASURE_UNCERTAINTY_DEFAULT,
                        "unit": MEASURE_UNIT_DEFAULT,
                        "measurand_id": meas_eid,
                        "metric_id": mid,
                        "observation_id": obs_id
                    })

        # (C) responses.csv -> observation per instance (store response text as metric)
        # attach Instance as elements too
        if "Instance" in resp_df.columns:
            for idx, row in resp_df.iterrows():
                instance = safe_str(row.get("Instance", f"row{idx}"))
                inst_eid = ensure_element("element", f"Instance={instance}", "Instance dimension", project_id)
                rows["evaluation_element"].append({"ref": inst_eid, "eval": eval_id})

                # measurand = instance itself
                meas_eid = inst_eid

                # dataset for responses: use __NO_TEMPLATE__ dataset
                dataset_id = dataset_ids_by_template.get("__NO_TEMPLATE__") or list(dataset_ids_by_template.values())[0]

                obs_id = stable_int(f"observation::{eval_id}::response::{instance}")
                rows["observation"].append({
                    "id": obs_id,
                    "observer": DEFAULT_OBSERVER,
                    "whenObserved": when_obs,
                    "tool_id": tool_id,
                    "dataset_id": dataset_id,
                    "eval_id": eval_id,
                    "name": "response",
                    "description": f"Raw responses | instance={instance}"
                })

                mid = metric_ids["response_text"]
                meas_id = stable_int(f"measure::{obs_id}::{mid}::{meas_eid}")
                rows["measure"].append({
                    "id": meas_id,
                    "value": safe_str(row.get("Response", ""))[:MEASURE_VALUE_MAXLEN],
                    "error": MEASURE_ERROR_DEFAULT,
                    "uncertainty": MEASURE_UNCERTAINTY_DEFAULT,
                    "unit": MEASURE_UNIT_DEFAULT,
                    "measurand_id": meas_eid,
                    "metric_id": mid,
                    "observation_id": obs_id
                })

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
        "evaluation": ["id"],
        "evaluation_element": ["ref", "eval"],
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
    # Example usage:
    # python accumulate_to_data_accumulated.py --repo . --manifest manifest.json --out data/data_accumulated
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
