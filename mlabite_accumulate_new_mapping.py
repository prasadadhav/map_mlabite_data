#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import io
import re
import hashlib
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

ERROR_STR = "ERROR"
TOOL_NAME = "MLABiTe"
DEFAULT_OBSERVER = "Test USER"
DATASET_LICENSING = "Open_Source"
DATASET_TYPE = "Test"
PROJECT_STATUS = "Ready"
EVAL_STATUS = "Done"
DATASHAPE_ACCEPTED_TARGET_VALUES = "1x1"
MEASURE_ERROR = "Not available for MLABiTe"
MEASURE_UNCERTAINTY = 0.0  # numeric column in DB; cannot store string "ERROR"

ROW_METRICS = ["Oracle Evaluation", "Oracle Prediction", "Evaluation"]
GLOBAL_METRICS = [
    "Passed Nr", "Failed Nr", "Error Nr", "Passed Pct",
    "Failed Pct", "Total", "Tolerance", "Tolerance Evaluation"
]
DIRECT_METRICS = [
    "Oracle Evaluation", "Oracle Prediction", "Evaluation",
    "Passed Nr", "Failed Nr", "Error Nr", "Total", "Tolerance"
]
DERIVED_METRICS = ["Tolerance Evaluation", "Passed Pct", "Failed Pct"]
METRIC_UNITS = {
    "Oracle Evaluation": "label",
    "Oracle Prediction": "label",
    "Evaluation": "label",
    "Passed Nr": "count",
    "Failed Nr": "count",
    "Error Nr": "count",
    "Passed Pct": "percent",
    "Failed Pct": "percent",
    "Total": "count",
    "Tolerance": "threshold",
    "Tolerance Evaluation": "label",
}


def stable_int(key: str) -> int:
    h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    return (int(h[:12], 16) % 2_000_000_000) + 1


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_str(value: Any) -> str:
    if value is None:
        return ERROR_STR
    if isinstance(value, float) and math.isnan(value):
        return ERROR_STR
    txt = str(value).strip()
    return txt if txt else ERROR_STR


def sniff_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    return ";" if sample.count(";") > sample.count(",") else ","


# def read_csv_flex(path: Path) -> pd.DataFrame:
#     sep = sniff_sep(path)
#     return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)

# Be tolerant of various CSV formatting issues by trying multiple parsing strategies
def read_csv_flex(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader for messy result files where:
    - separator may be ',' or ';'
    - line endings may vary (\n, \r\n, \r)
    - quoting may be inconsistent
    - some rows may be malformed

    Strategy:
    1. Read raw text and normalize line endings
    2. Try csv.Sniffer to detect delimiter
    3. Try pandas with detected delimiter using python engine
    4. Fallback to ',' and ';'
    5. Final fallback: skip bad lines
    6. Return all values as strings, replace missing with ERROR
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Normalize line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Remove UTF-8 BOM if present
    raw = raw.lstrip("\ufeff")

    # Remove completely empty lines
    lines = [line for line in raw.split("\n") if line.strip() != ""]
    cleaned = "\n".join(lines)

    # Helper to read from in-memory text
    def _try_read(text: str, sep: str, engine: str = "python", on_bad_lines="error"):
        return pd.read_csv(
            io.StringIO(text),
            sep=sep,
            dtype=str,
            keep_default_na=False,
            engine=engine,
            quotechar='"',
            doublequote=True,
            on_bad_lines=on_bad_lines
        )

    # 1. Try csv.Sniffer
    candidate_seps = []
    try:
        sample = "\n".join(lines[:10])
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        candidate_seps.append(dialect.delimiter)
    except Exception:
        pass

    # 2. Add manual fallbacks
    for sep in [",", ";"]:
        if sep not in candidate_seps:
            candidate_seps.append(sep)

    # 3. Strict attempts first
    errors = []
    for sep in candidate_seps:
        try:
            df = _try_read(cleaned, sep=sep, engine="python", on_bad_lines="error")
            return df.fillna(ERROR_STR).replace("", ERROR_STR)
        except Exception as e:
            errors.append(f"sep={sep}, engine=python, strict: {e}")

    # 4. Last fallback: skip malformed rows
    for sep in candidate_seps:
        try:
            df = _try_read(cleaned, sep=sep, engine="python", on_bad_lines="skip")
            return df.fillna(ERROR_STR).replace("", ERROR_STR)
        except Exception as e:
            errors.append(f"sep={sep}, engine=python, skip_bad_lines: {e}")

    raise ValueError(
        f"Could not parse CSV file: {path}\n"
        + "\n".join(errors)
    )


def parse_when_observed(timestamp_dir: str) -> str:
    try:
        return datetime.strptime(timestamp_dir, "%Y%m%d_%H%M%S").isoformat(sep=" ")
    except Exception:
        return ERROR_STR


def load_manifest(repo_root: Path, manifest_path: Path) -> dict:
    candidates = [
        repo_root / manifest_path,
        repo_root / "data" / "data_accumulated" / manifest_path,
        manifest_path,
    ]
    for cand in candidates:
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Could not find manifest.json. Tried: {candidates}")


def infer_project_name(repo_root: Path, run: dict) -> str:
    evals_path = Path(run["paths"]["evals_csv"])
    if not evals_path.is_absolute():
        evals_path = repo_root / evals_path
    try:
        rel = evals_path.resolve().relative_to((repo_root / "data").resolve())
        parts = rel.parts
        if parts and parts[0] != "data_accumulated":
            return parts[0]
    except Exception:
        pass
    return safe_str(run.get("test_name") or ERROR_STR)


def canonical_template_key(text: str) -> str:
    t = safe_str(text)
    t = re.sub(r"\s+", " ", t).strip().lower()
    t = re.sub(r"\{[^}]+\}", "{}", t)
    return t


def template_regex(template: str) -> re.Pattern:
    escaped = re.escape(safe_str(template))
    escaped = re.sub(r"\\\{[^}]+\\\}", r".+?", escaped)
    escaped = re.sub(r"\\\s\+", r"\\s+", escaped)
    return re.compile(rf"^{escaped}$", re.IGNORECASE | re.DOTALL)


def count_placeholders(template: str) -> int:
    return len(re.findall(r"\{[^}]+\}", safe_str(template)))


def count_matches_in_instance(template: str, instance: str) -> int:
    names = re.findall(r"\{([^}]+)\}", safe_str(template))
    score = 0
    for n in names:
        token = n.strip()
        if token and token.lower() in safe_str(instance).lower():
            score += 1
    return score


def flatten_json(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else k
            yield from flatten_json(v, name)
    elif isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            yield prefix, json.dumps(obj, ensure_ascii=False), type(obj).__name__
        else:
            for i, v in enumerate(obj):
                name = f"{prefix}[{i}]"
                yield from flatten_json(v, name)
    else:
        yield prefix, obj, type(obj).__name__


def extract_config_communities(config: dict, concern: str, language: str) -> List[dict]:
    """
    Best-effort extractor for MLABiTe community substitutions.
    Returns a list of dicts like {AGE:..., GENDER1:..., GENDER2:...}.
    """
    concern_l = safe_str(concern).lower()
    lang_l = safe_str(language).lower()
    hits: List[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            lowered = {str(k).lower(): v for k, v in node.items()}
            maybe_concern = safe_str(lowered.get("concern", "")).lower()
            maybe_langs = lowered.get("languages") or lowered.get("language")
            maybe_lang_match = False
            if isinstance(maybe_langs, list):
                maybe_lang_match = any(lang_l == safe_str(x).lower() for x in maybe_langs)
            elif maybe_langs is not None:
                maybe_lang_match = lang_l == safe_str(maybe_langs).lower()

            if maybe_concern == concern_l or maybe_lang_match:
                # direct list of substitution dicts
                for k, v in node.items():
                    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                        for item in v:
                            upper_keys = {str(kk).upper(): vv for kk, vv in item.items()}
                            if upper_keys:
                                hits.append(upper_keys)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    uniq: List[dict] = []
    seen = set()
    for h in hits:
        key = json.dumps(h, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq


def estimate_instance_count(template: str, config: dict, concern: str, language: str) -> int:
    placeholders = re.findall(r"\{([^}]+)\}", safe_str(template))
    if not placeholders:
        return 1
    community_maps = extract_config_communities(config, concern, language)
    if not community_maps:
        return max(1, count_placeholders(template))
    valid = 0
    for item in community_maps:
        if any(p.strip().upper() in item for p in placeholders):
            valid += 1
    return max(1, valid or len(community_maps))


def substitute_template(template: str, mapping: dict) -> str:
    out = safe_str(template)
    for k, v in mapping.items():
        out = out.replace("{" + str(k) + "}", safe_str(v))
    return out


def assign_instances_to_template(
    template: str,
    concern: str,
    language: str,
    config: dict,
    candidate_rows: List[dict],
) -> List[dict]:
    """
    Two-step matching:
    1) estimate count from config.json
    2) rank candidates using regex and placeholder substitution similarity
    """
    expected = estimate_instance_count(template, config, concern, language)
    regex = template_regex(template)
    community_maps = extract_config_communities(config, concern, language)
    rendered = [substitute_template(template, m) for m in community_maps] if community_maps else []

    scored = []
    for row in candidate_rows:
        instance = safe_str(row.get("Instance"))
        score = 0
        if regex.match(instance):
            score += 100
        score += count_matches_in_instance(template, instance)
        if any(instance.strip().lower() == r.strip().lower() for r in rendered):
            score += 1000
        scored.append((score, row))

    scored.sort(key=lambda x: (-x[0], candidate_rows.index(x[1])))
    strong = [r for s, r in scored if s > 0]
    if strong:
        return strong[:expected]
    return candidate_rows[:expected]


def build_metric_description(name: str) -> str:
    return f"Auto-generated description for metric '{name}'."


def build_direct_description(name: str) -> str:
    return f"Auto-generated description for direct metric '{name}'."


def build_derived_description(name: str) -> str:
    return f"Auto-generated description for derived metric '{name}'."


def build_metriccategory_description(name: str) -> str:
    return f"Auto-generated description for metric category '{name}'."


def ensure_columns(df: pd.DataFrame, required: List[str], src: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {src}")
    
# helper for model
def load_model_from_global(glob_df: pd.DataFrame, global_csv: Path) -> str:
    if "Model" not in glob_df.columns:
        raise ValueError(f"Missing column ['Model'] in {global_csv}")
    if glob_df.empty:
        return ERROR_STR
    model_name = safe_str(glob_df.iloc[0].get("Model"))
    return model_name


def build_model_description(model_name: str) -> str:
    return f"Model used for this MLABiTe evaluation: {model_name}"


def ensure_model_row(
    rows: Dict[str, List[dict]],
    seen_model_ids: set,
    model_name: str,
    eval_id: int,
    project_name: str,
    timestamp_dir: str,
) -> int:
    """
    One model per evaluation.
    The model row is also an Element child in the DB schema, so model.id must exist in element.id too.
    """
    model_id = stable_int(f"model::{project_name}::{timestamp_dir}::{eval_id}::{model_name}")

    if model_id not in seen_model_ids:
        seen_model_ids.add(model_id)

        rows["element"].append({
            "id": model_id,
            "name": model_name,
            "description": build_model_description(model_name),
            "project_id": ERROR_STR if False else None,   # use None for nullable FK
            "type_spec": "model",
        })

        rows["model"].append({
            "id": model_id,
            "name": model_name,
            "description": build_model_description(model_name),
            "pid": model_name,
            "data": ERROR_STR,
            "source": "MLABiTe_global_evaluation",
            "licensing": DATASET_LICENSING,
        })

    return model_id


def load_tool_id(tool_csv: Path) -> int:
    if not tool_csv.exists():
        raise FileNotFoundError(
            f"tool.csv not found at {tool_csv}. Please create it manually as planned before running the accumulator."
        )
    df = read_csv_flex(tool_csv)
    ensure_columns(df, ["id"], tool_csv)
    if "name" in df.columns:
        hits = df[df["name"].astype(str).str.lower() == TOOL_NAME.lower()]
        if not hits.empty:
            return int(hits.iloc[0]["id"])
    return int(df.iloc[0]["id"])


def append_unique_csv(out_csv: Path, df: pd.DataFrame, subset: List[str]) -> None:
    if out_csv.exists():
        old = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=subset, keep="last")
        merged.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False)


def main(repo_root: Path, manifest_path: Path, out_dir: Path, tool_csv: Path) -> None:
    manifest = load_manifest(repo_root, manifest_path)
    runs = manifest.get("runs", [])
    ensure_dir(out_dir)

    tool_id = load_tool_id(tool_csv if tool_csv.is_absolute() else repo_root / tool_csv)
    datashape_id = stable_int("datashape::1x1")

    # to keep track of unique datasets (prompt templates)
    seen_dataset_ids = set()
    seen_dataset_keys = set()

    # track model
    seen_model_ids = set()
    seen_model_dataset_links = set()

    rows: Dict[str, List[dict]] = defaultdict(list)
    rows["datashape"].append({
        "id": datashape_id,
        "accepted_target_values": DATASHAPE_ACCEPTED_TARGET_VALUES,
    })

    # Metrics and metric subtypes
    all_metrics = ROW_METRICS + GLOBAL_METRICS
    for name in all_metrics:
        metric_id = stable_int(f"metric::{name}")
        type_spec = "Derived" if name in DERIVED_METRICS else "Direct"
        rows["metric"].append({
            "id": metric_id,
            "type_spec": type_spec,
            "name": name,
            "description": build_metric_description(name),
        })
        if name in DIRECT_METRICS:
            rows["direct"].append({
                "id": metric_id,
                "name": name,
                "description": build_direct_description(name),
            })
        if name in DERIVED_METRICS:
            rows["derived"].append({
                "id": metric_id,
                "name": name,
                "description": build_derived_description(name),
                "expression": "To Do",
            })

    # derived_metric FK table per requested mapping
    for base, derived in [
        ("Passed Nr", "Passed Pct"),
        ("Total", "Passed Pct"),
        ("Failed Nr", "Failed Pct"),
        ("Total", "Failed Pct"),
    ]:
        rows["derived_metric"].append({
            "baseMetric": stable_int(f"metric::{base}"),
            "derivedBy": stable_int(f"metric::{derived}"),
        })

    seen_metric_categories = set()

    for run in runs:
        timestamp_dir = safe_str(run.get("timestamp_dir"))
        when_observed = parse_when_observed(timestamp_dir)
        project_name = infer_project_name(repo_root, run)
        project_id = stable_int(f"project::{project_name}")
        rows["project"].append({
            "id": project_id,
            "name": project_name,
            "status": PROJECT_STATUS,
        })

        paths = run["paths"]
        evals_csv = Path(paths["evals_csv"])
        global_csv = Path(paths["global_csv"])
        responses_csv = Path(paths["responses_csv"])
        config_json = Path(paths["config_json"])
        if not evals_csv.is_absolute():
            evals_csv = repo_root / evals_csv
        if not global_csv.is_absolute():
            global_csv = repo_root / global_csv
        if not responses_csv.is_absolute():
            responses_csv = repo_root / responses_csv
        if not config_json.is_absolute():
            config_json = repo_root / config_json

        eval_df = read_csv_flex(evals_csv)
        glob_df = read_csv_flex(global_csv)
        resp_df = read_csv_flex(responses_csv)
        config = json.loads(config_json.read_text(encoding="utf-8"))

        ensure_columns(eval_df, [
            "Concern", "Language", "Template", "Oracle Evaluation",
            "Oracle Prediction", "Evaluation"
        ], evals_csv)
        ensure_columns(resp_df, ["Instance", "Response"], responses_csv)
        ensure_columns(glob_df, GLOBAL_METRICS, global_csv)

        language = safe_str(eval_df.iloc[0].get("Language") if not eval_df.empty else run.get("language"))
        concern_values = list(dict.fromkeys(eval_df["Concern"].astype(str).tolist()))

        config_id = stable_int(f"configuration::{config_json.as_posix()}")
        rows["configuration"].append({
            "id": config_id,
            "name": str(config_json),
            "description": f"{project_name}_{when_observed}_{language}",
        })

        for name, value, typ in flatten_json(config):
            confparam_id = stable_int(f"confparam::{config_id}::{name}")
            rows["confparam"].append({
                "id": confparam_id,
                "name": safe_str(name),
                "description": safe_str(name),
                "param_type": safe_str(typ),
                "value": safe_str(value if not isinstance(value, str) else value),
                "conf_id": config_id,
            })

        eval_id = stable_int(f"evaluation::{project_name}::{timestamp_dir}::{language}")
        rows["evaluation"].append({
            "id": eval_id,
            "status": EVAL_STATUS,
            "config_id": config_id,
            "project_id": project_id,
        })

        global_values = glob_df.iloc[0].to_dict() if not glob_df.empty else {m: ERROR_STR for m in GLOBAL_METRICS}

        # model
        model_name = load_model_from_global(glob_df, global_csv)
        model_id = ensure_model_row(
            rows=rows,
            seen_model_ids=seen_model_ids,
            model_name=model_name,
            eval_id=eval_id,
            project_name=project_name,
            timestamp_dir=timestamp_dir,
        )

        # create metric categories by concern and connect them to all metrics
        for concern in concern_values:
            mc_name = safe_str(concern)
            if mc_name not in seen_metric_categories:
                seen_metric_categories.add(mc_name)
                mc_id = stable_int(f"metriccategory::{mc_name}")
                rows["metriccategory"].append({
                    "id": mc_id,
                    "name": mc_name,
                    "description": build_metriccategory_description(mc_name),
                })
                for metric_name in all_metrics:
                    rows["metriccategory_metric"].append({
                        "category": mc_id,
                        "metrics": stable_int(f"metric::{metric_name}"),
                    })

        # Group responses by original order; maintain unused pool
        response_rows = resp_df.to_dict(orient="records")
        used_response_ids = set()

        unique_templates = eval_df[["Concern", "Language", "Template"]].drop_duplicates().to_dict(orient="records")
        template_to_response_rows: Dict[Tuple[str, str, str], List[dict]] = {}

        # Pass 1: assign best candidates to each template
        for tpl in unique_templates:
            concern = safe_str(tpl["Concern"])
            lang = safe_str(tpl["Language"])
            template = safe_str(tpl["Template"])
            candidates = [r for idx, r in enumerate(response_rows) if idx not in used_response_ids]
            selected = assign_instances_to_template(template, concern, lang, config, candidates)
            selected_indices = []
            for row in selected:
                idx = response_rows.index(row)
                if idx not in used_response_ids:
                    used_response_ids.add(idx)
                    selected_indices.append(idx)
            # If selection collapsed because dict equality hit duplicates, recover by sequential fallback
            if not selected_indices:
                expected = estimate_instance_count(template, config, concern, lang)
                for idx, row in enumerate(response_rows):
                    if idx in used_response_ids:
                        continue
                    selected_indices.append(idx)
                    used_response_ids.add(idx)
                    if len(selected_indices) >= expected:
                        break
            template_to_response_rows[(concern, lang, template)] = [response_rows[i] for i in selected_indices]

        # Pass 2: create dataset -> element -> observation -> measures chain
        # only add unique prompt templates as datasets
        
        for eval_row_idx, eval_row in eval_df.iterrows():
            concern = safe_str(eval_row.get("Concern"))
            lang = safe_str(eval_row.get("Language"))
            template = safe_str(eval_row.get("Template"))

            # dataset_id = stable_int(f"dataset::{project_name}::{concern}::{lang}::{canonical_template_key(template)}")
            # rows["dataset"].append({
            #     "id": dataset_id,
            #     "name": concern,
            #     "description": template,
            #     "source": ERROR_STR,
            #     "version": lang,
            #     "licensing": DATASET_LICENSING,
            #     "dataset_type": DATASET_TYPE,
            #     "datashape_id": datashape_id,
            # })
            # dataset uniqueness should be based only on the prompt template text
            # dataset_id = stable_int(f"dataset::{canonical_template_key(template)}")

            # only keep unique templates
            # dataset_key = canonical_template_key(template)
            # dataset_id = stable_int(f"dataset::{dataset_key}")
            # if dataset_key not in seen_dataset_keys:
            # # if dataset_id not in seen_dataset_ids:
            #     rows["dataset"].append({
            #         "id": dataset_id,
            #         "name": concern,
            #         "description": template,
            #         "source": ERROR_STR,
            #         "version": lang,
            #         "licensing": DATASET_LICENSING,
            #         "dataset_type": DATASET_TYPE,
            #         "datashape_id": datashape_id,
            #     })
            #     seen_dataset_ids.add(dataset_id)
            #     seen_dataset_keys.add(dataset_key)
            dataset_key = canonical_template_key(template)
            dataset_id = stable_int(f"dataset::{dataset_key}")

            if dataset_key not in seen_dataset_keys:
                rows["dataset"].append({
                    "id": dataset_id,
                    "name": concern,
                    "description": template,
                    "source": ERROR_STR,
                    "version": lang,
                    "licensing": DATASET_LICENSING,
                    "dataset_type": DATASET_TYPE,
                    "datashape_id": datashape_id,
                })
                seen_dataset_ids.add(dataset_id)
                seen_dataset_keys.add(dataset_key)

            # link model <-> dataset for this evaluation
            md_key = (model_id, dataset_id)
            if md_key not in seen_model_dataset_links:
                rows["model_dataset"].append({
                    "models": model_id,
                    "dataset": dataset_id,
                })
                seen_model_dataset_links.add(md_key)

            # rows["dataset"].append({
            #     "id": dataset_id,
            #     "name": concern,
            #     "description": template,
            #     "source": ERROR_STR,
            #     "version": lang,
            #     "licensing": DATASET_LICENSING,
            #     "dataset_type": DATASET_TYPE,
            #     "datashape_id": datashape_id,
            # })

            linked_responses = template_to_response_rows.get((concern, lang, template), [])
            if not linked_responses:
                linked_responses = [{"Instance": ERROR_STR, "Response": ERROR_STR}]

            for inst_pos, resp_row in enumerate(linked_responses, start=1):
                instance_text = safe_str(resp_row.get("Instance"))
                response_text = safe_str(resp_row.get("Response"))
                element_id = stable_int(
                    f"element::{dataset_id}::{inst_pos}::{hashlib.sha1(instance_text.encode('utf-8', errors='ignore')).hexdigest()[:12]}"
                )
                rows["element"].append({
                    "id": element_id,
                    "name": str(dataset_id),
                    "description": instance_text,
                    "project_id": project_id,
                    "type_spec": f"{concern}_{lang}",
                })
                rows["evaluation_element"].append({
                    "eval": eval_id,
                    "ref": element_id,
                })

                rows["evaluates_eval"].append({
                    "evaluates": element_id,
                    "evalu": eval_id,
                })

                observation_id = stable_int(
                    f"observation::{eval_id}::{tool_id}::{dataset_id}::{element_id}::{timestamp_dir}"
                )
                rows["observation"].append({
                    "id": observation_id,
                    "name": f"{concern}_{lang}_{timestamp_dir}",
                    "description": response_text,
                    "observer": DEFAULT_OBSERVER,
                    "whenObserved": when_observed,
                    "eval_id": eval_id,
                    "tool_id": tool_id,
                    "dataset_id": dataset_id,
                })

                # Row-level metrics replicated per observation
                for metric_name in ROW_METRICS:
                    metric_id = stable_int(f"metric::{metric_name}")
                    value = safe_str(eval_row.get(metric_name))
                    measure_id = stable_int(f"measure::{observation_id}::{metric_id}::{element_id}")
                    rows["measure"].append({
                        "id": measure_id,
                        "value": value,
                        "error": MEASURE_ERROR,
                        "uncertainty": MEASURE_UNCERTAINTY,
                        "unit": METRIC_UNITS.get(metric_name, ERROR_STR),
                        "observation_id": observation_id,
                        "metric_id": metric_id,
                        "measurand_id": element_id,
                    })

                # Global metrics replicated per observation to avoid NAs/ERROR rows in measure
                for metric_name in GLOBAL_METRICS:
                    metric_id = stable_int(f"metric::{metric_name}")
                    value = safe_str(global_values.get(metric_name))
                    measure_id = stable_int(f"measure::{observation_id}::{metric_id}::{element_id}")
                    rows["measure"].append({
                        "id": measure_id,
                        "value": value,
                        "error": MEASURE_ERROR,
                        "uncertainty": MEASURE_UNCERTAINTY,
                        "unit": METRIC_UNITS.get(metric_name, ERROR_STR),
                        "observation_id": observation_id,
                        "metric_id": metric_id,
                        "measurand_id": element_id,
                    })

    # Write CSVs
    key_map = {
        "project": ["id"],
        "datashape": ["id"],
        "configuration": ["id"],
        "confparam": ["id"],
        "evaluation": ["id"],
        "dataset": ["id"],
        "element": ["id"],
        "model": ["id"],
        "model_dataset": ["models", "dataset"],
        "evaluation_element": ["eval", "ref"],
        "evaluates_eval": ["evaluates", "evalu"],
        "observation": ["id"],
        "metric": ["id"],
        "direct": ["id"],
        "derived": ["id"],
        "derived_metric": ["baseMetric", "derivedBy"],
        "metriccategory": ["id"],
        "metriccategory_metric": ["category", "metrics"],
        "measure": ["id"],
    }
    for table, table_rows in rows.items():
        if not table_rows:
            continue
        df = pd.DataFrame(table_rows)
        out_csv = out_dir / f"{table}.csv"
        append_unique_csv(out_csv, df, key_map[table])
        print(f"Wrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Accumulate MLABiTe data into normalized CSVs for the refactored DB schema.")
    ap.add_argument("--repo", default=".", help="Repository root")
    ap.add_argument("--manifest", default="manifest.json", help="Path to manifest.json (repo root or data/data_accumulated)")
    ap.add_argument("--out", default="data/data_accumulated", help="Output directory for accumulated CSVs")
    ap.add_argument("--tool-csv", default="data/data_accumulated/tool.csv", help="Manual tool.csv path")
    args = ap.parse_args()
    main(Path(args.repo), Path(args.manifest), Path(args.out), Path(args.tool_csv))
