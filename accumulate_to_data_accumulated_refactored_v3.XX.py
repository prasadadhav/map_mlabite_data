#!/usr/bin/env python3
"""
MLABiTe accumulator (v3.xx)

Design goals:
- Keep evaluation-driven measure creation (like v3) to avoid NA cascades.
- Build template datasets from *_evaluations.csv:
    dataset.source  = Template
    dataset.version = Language
    dataset_type    = "test"
    dataset.id      = auto (stable deterministic)
- Assign response Instances (filled templates) from *_responses.csv to templates via regex derived from Template text.
- Create one template_instance element per matched Instance:
    element.name        = str(dataset.id)  (links instance -> template dataset)
    element.description = Instance         (filled prompt)
- Create one observation per template instance:
    observation.dataset_id  = dataset.id
    observation.description = Response     (LLM output)
- Attach measures (evaluation/oracle_*) using evaluation rows only.
  If instances > evaluation rows for a template, duplicate measures across extra instances (modulo mapping).

Notes:
- Measures come ONLY from evaluations.csv rows.
- Observations/elements are emitted per matched instance (bounded by config communities hint when available).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

NA_STR = "NA"

DEFAULT_OBSERVER = "MLABiTe"
TOOL_NAME = "MLABiTe"
TOOL_SOURCE = "LIST"
TOOL_LICENSING = "Open_Source"

PROJECT_STATUS = "Ready"
EVALUATION_STATUS = "Done"

DATASET_LICENSING_DEFAULT = "Proprietary"
DATASET_TYPE_DEFAULT = "test"

MEASURE_ERROR_DEFAULT = "Not measured"
MEASURE_UNIT_DEFAULT = "Not defined"
MEASURE_UNCERTAINTY_DEFAULT = 0.0

OBS_DESC_MAXLEN = 10000
ELEMENT_DESC_MAXLEN = 10000
DATASET_SOURCE_MAXLEN = 10000
MEASURE_VALUE_MAXLEN = 10000

DEFAULT_PROJECT_NAME = "default_project"

DERIVED_METRIC_NAMES = {"Passed Pct", "Failed Pct", "Tolerance Evaluation"}


def stable_int(key: str) -> int:
    h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    n = int(h[:8], 16)
    return (n % 2_000_000_000) + 1


def parse_timestamp_dir(ts: str) -> str:
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.isoformat(sep=" ")
    except Exception:
        return datetime.utcnow().isoformat(sep=" ")


def norm_path(p: str) -> Path:
    return Path(p.replace("\\", os.sep))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sniff_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    return ";" if sample.count(";") > sample.count(",") else ","


def read_csv_flex(path: Path) -> pd.DataFrame:
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
    """Robust parsing where Response may contain delimiter."""
    sep = sniff_sep(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return pd.DataFrame(columns=["Provider", "Model", "Instance", "Response"])
    rows: List[Dict[str, str]] = []
    for raw in lines[1:]:
        if not raw.strip():
            continue
        parts = raw.split(sep)
        provider = parts[0].strip() if len(parts) > 0 else ""
        model = parts[1].strip() if len(parts) > 1 else ""
        instance = parts[2].strip() if len(parts) > 2 else ""
        response = sep.join(parts[3:]).strip() if len(parts) > 3 else ""
        rows.append({"Provider": provider, "Model": model, "Instance": instance, "Response": response})
    return pd.DataFrame(rows, columns=["Provider", "Model", "Instance", "Response"])


def safe_required(x: Any) -> str:
    if x is None:
        return NA_STR
    if isinstance(x, float) and pd.isna(x):
        return NA_STR
    s = str(x)
    return s if s != "" else NA_STR


def safe_optional(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def normalize_metric_category_from_test_name(test_name: str) -> str:
    t = (test_name or "").strip()
    if t.lower().startswith("test-"):
        return t[5:]
    return t or "unknown_category"


def normalize_metric_category_from_concern(concern: str) -> str:
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
    t = (template or "").strip()
    if not t:
        return "unknown_template"
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()[:16]


def build_template_regex(template: str) -> re.Pattern:
    t = (template or "").strip()
    if not t:
        return re.compile(r"^$")
    if "{" not in t or "}" not in t:
        return re.compile(r"^" + re.escape(t) + r"$")
    out = []
    i = 0
    while i < len(t):
        if t[i] == "{":
            j = t.find("}", i + 1)
            if j == -1:
                out.append(re.escape(t[i]))
                i += 1
            else:
                out.append(r"(.+?)")
                i = j + 1
        else:
            out.append(re.escape(t[i]))
            i += 1
    return re.compile(r"^" + "".join(out) + r"$", flags=re.DOTALL)


def score_template_match(template: str, instance: str) -> int:
    t = (template or "").strip()
    inst = (instance or "").strip()
    if not t or not inst:
        return 0
    static = re.sub(r"\{[^}]+\}", " ", t)
    static = re.sub(r"\s+", " ", static).strip()
    if not static:
        return 1
    tokens = [tok for tok in re.split(r"\W+", static) if tok]
    if not tokens:
        return 1
    inst_low = inst.lower()
    return sum(1 for tok in tokens if tok.lower() in inst_low)


def get_communities_for_concern(cfg: dict, concern: str) -> List[str]:
    c = (concern or "").strip()
    if not c:
        return []
    norm_c = normalize_metric_category_from_concern(c)

    if "communities" in cfg and isinstance(cfg["communities"], dict):
        for key, vals in cfg["communities"].items():
            if normalize_metric_category_from_concern(str(key)) == norm_c:
                if isinstance(vals, list):
                    return [str(x) for x in vals]
                if isinstance(vals, dict):
                    return [str(x) for x in vals.keys()]

    for topk in ["tests", "testCases", "scenarios", "concerns"]:
        if topk in cfg and isinstance(cfg[topk], dict):
            for key, obj in cfg[topk].items():
                if normalize_metric_category_from_concern(str(key)) == norm_c:
                    if isinstance(obj, dict) and "communities" in obj:
                        vals = obj["communities"]
                        if isinstance(vals, list):
                            return [str(x) for x in vals]
                        if isinstance(vals, dict):
                            return [str(x) for x in vals.keys()]
    return []


def upsert_append(out_csv: Path, df: pd.DataFrame, key_cols: List[str]) -> None:
    if out_csv.exists():
        prev = read_csv_flex(out_csv)
        merged = pd.concat([prev, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="first")
        merged.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False)


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
        "evaluation",
        "evaluation_element",
        "evaluates_eval",
        "metric",
        "direct",
        "derived",
        "observation",
        "measure",
    ]}

    tool_id = stable_int(f"tool::{TOOL_NAME}")
    rows["tool"].append({
        "id": tool_id,
        "source": TOOL_SOURCE,
        "version": "v3.xx",
        "name": TOOL_NAME,
        "licensing": TOOL_LICENSING,
    })

    metric_ids: Dict[str, int] = {}

    def ensure_metric(metric_name: str, description: str = "") -> int:
        key = metric_name.strip()
        if key in metric_ids:
            return metric_ids[key]
        mid = stable_int(f"metric::{key}")
        metric_ids[key] = mid
        if key in DERIVED_METRIC_NAMES:
            rows["metric"].append({"id": mid, "type_spec": "Derived", "name": key, "description": (description or key)[:100]})
            rows["derived"].append({"id": mid, "expression": NA_STR})
        else:
            rows["metric"].append({"id": mid, "type_spec": "Direct", "name": key, "description": (description or key)[:100]})
            rows["direct"].append({"id": mid})
        return mid

    ensure_metric("evaluation", "Row-level evaluation result")
    ensure_metric("oracle_prediction", "Row-level oracle prediction")
    ensure_metric("oracle_evaluation", "Row-level oracle evaluation")

    element_seen: Set[int] = set()

    def ensure_element(type_spec: str, name: str, description: str, project_id: Optional[int]) -> int:
        desc_hash = hashlib.sha1((description or "").encode("utf-8", errors="ignore")).hexdigest()[:10]
        eid = stable_int(f"element::{type_spec}::{name}::{desc_hash}")
        if eid in element_seen:
            return eid
        element_seen.add(eid)
        d = (description or NA_STR).strip()
        if len(d) > ELEMENT_DESC_MAXLEN:
            d = d[:ELEMENT_DESC_MAXLEN]
        rows["element"].append({
            "id": eid,
            "project_id": project_id,
            "type_spec": safe_required(type_spec)[:50],
            "name": safe_required(name)[:100],
            "description": safe_required(d),
        })
        return eid

    datashape_ids: Dict[str, int] = {}

    def ensure_datashape(project_name: str, metric_category: str) -> int:
        key = f"{project_name}/{metric_category}"
        if key in datashape_ids:
            return datashape_ids[key]
        ds_id = stable_int(f"datashape::{key}")
        datashape_ids[key] = ds_id
        rows["datashape"].append({"id": ds_id, "accepted_target_values": safe_required(key)[:100]})
        return ds_id

    dataset_seen: Set[int] = set()

    def ensure_template_dataset(project_name: str, metric_category: str, language_from_row: str, template_text: str) -> int:
        ds_shape_id = ensure_datashape(project_name, metric_category)
        tkey = template_key(template_text)
        dataset_id = stable_int(f"dataset::{project_name}/{metric_category}::{language_from_row}::{tkey}")
        if dataset_id not in dataset_seen:
            dataset_seen.add(dataset_id)
            tmpl = (template_text or NA_STR).strip()
            if len(tmpl) > DATASET_SOURCE_MAXLEN:
                tmpl = tmpl[:DATASET_SOURCE_MAXLEN]
            rows["dataset"].append({
                "id": dataset_id,
                "source": safe_required(tmpl),
                "version": safe_required(language_from_row)[:100],
                "licensing": DATASET_LICENSING_DEFAULT,
                "dataset_type": DATASET_TYPE_DEFAULT,
                "datashape_id": ds_shape_id,
            })
        return dataset_id

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
            "name": safe_required(f"config_{project_name}_{metric_category}_{sig}")[:100],
            "description": safe_required(f"Config for {project_name}/{metric_category} (sig={sig})")[:100],
        })
        for k, v in cfg.items():
            cp_id = stable_int(f"confparam::{key}::{k}")
            vv = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            rows["confparam"].append({
                "id": cp_id,
                "param_type": "json",
                "value": safe_required(vv)[:MEASURE_VALUE_MAXLEN],
                "conf_id": config_id,
                "name": safe_required(k)[:100],
                "description": "from config.json",
            })
        return config_id, sig

    eval_registry: Dict[str, int] = {}

    def get_or_create_evaluation(project_id: int, project_name: str, metric_category: str, language: str, cfg_sig: str, config_id: int) -> int:
        k = f"{project_name}::{metric_category}::{language}::{cfg_sig}"
        if k in eval_registry:
            return eval_registry[k]
        eval_id = stable_int(f"evaluation::{k}")
        eval_registry[k] = eval_id
        rows["evaluation"].append({"id": eval_id, "status": EVALUATION_STATUS, "config_id": config_id, "project_id": project_id})
        return eval_id

    def add_measure(obs_id: int, measurand_eid: int, metric_name: str, value: Any) -> None:
        mid = metric_ids[metric_name]
        meas_id = stable_int(f"measure::{obs_id}::{mid}::{measurand_eid}")
        v = safe_required(value)
        if len(v) > MEASURE_VALUE_MAXLEN:
            v = v[:MEASURE_VALUE_MAXLEN]
        rows["measure"].append({
            "id": meas_id,
            "value": v,
            "error": safe_required(MEASURE_ERROR_DEFAULT)[:100],
            "uncertainty": float(MEASURE_UNCERTAINTY_DEFAULT),
            "unit": safe_required(MEASURE_UNIT_DEFAULT)[:100],
            "measurand_id": measurand_eid,
            "metric_id": mid,
            "observation_id": obs_id,
        })

    for run in runs:
        ts = run.get("timestamp_dir") or "unknown_ts"
        test_name = run.get("test_name") or "unknown_test"
        language_dir = run.get("language") or "unknown_lang"
        paths = run["paths"]

        project_name = run.get("provider_family") or DEFAULT_PROJECT_NAME
        project_id = stable_int(f"project::{project_name}")
        rows["project"].append({"id": project_id, "name": safe_required(project_name)[:100], "status": PROJECT_STATUS})

        evals_df = read_csv_flex(repo_root / norm_path(paths["evals_csv"]))
        global_df = read_csv_flex(repo_root / norm_path(paths["global_csv"]))
        cfg = json.loads((repo_root / norm_path(paths["config_json"])).read_text(encoding="utf-8"))

        resp_path = repo_root / norm_path(paths["responses_csv"])
        try:
            resp_df = read_csv_flex(resp_path)
        except Exception:
            resp_df = read_responses_csv_robust(resp_path)
        if not {"Model", "Instance", "Response"}.issubset(set(resp_df.columns)):
            resp_df = read_responses_csv_robust(resp_path)

        fallback_category = normalize_metric_category_from_test_name(test_name)
        categories: Set[str] = set()
        for df in (evals_df, global_df):
            if "Concern" in df.columns:
                for v in df["Concern"]:
                    cat = normalize_metric_category_from_concern(safe_optional(v))
                    if cat:
                        categories.add(cat)
        if not categories:
            categories = {fallback_category}

        resp_by_model: Dict[str, pd.DataFrame] = {}
        if "Model" in resp_df.columns:
            for m, g in resp_df.groupby("Model", dropna=False):
                resp_by_model[safe_optional(m)] = g.reset_index(drop=True).copy()

        when_obs = parse_timestamp_dir(ts)

        for metric_category in sorted(categories):
            if "Concern" in evals_df.columns:
                eval_slice = evals_df[evals_df["Concern"].astype(str).apply(
                    lambda x: normalize_metric_category_from_concern(x) == metric_category
                )].copy()
            else:
                eval_slice = evals_df.copy()

            if len(eval_slice) == 0 or "Model" not in eval_slice.columns:
                continue

            communities = get_communities_for_concern(cfg, metric_category)
            max_instances_hint = max(1, len(communities)) if communities else 10**9

            config_id, cfg_sig = get_or_create_configuration(project_name, metric_category, cfg)
            eval_id = get_or_create_evaluation(project_id, project_name, metric_category, language_dir, cfg_sig, config_id)

            for model_name, model_evals in eval_slice.groupby("Model", dropna=False):
                model_name = safe_optional(model_name)
                model_evals = model_evals.reset_index(drop=True).copy()

                resp_m = resp_by_model.get(model_name)
                resp_rows: List[Dict[str, str]] = []
                if resp_m is not None:
                    for _, rr in resp_m.iterrows():
                        inst = safe_optional(rr.get("Instance")).strip()
                        resp = safe_optional(rr.get("Response")).strip()
                        if inst:
                            resp_rows.append({"Instance": inst, "Response": resp})

                if "Template" not in model_evals.columns:
                    continue

                # Group evaluation rows by template text
                tmpl_groups: Dict[str, pd.DataFrame] = {}
                for tmpl_text, g in model_evals.groupby("Template", dropna=False):
                    tmpl_groups[safe_optional(tmpl_text)] = g.reset_index(drop=True).copy()

                # Build per-template metadata keyed by dataset.id
                tmpl_meta: Dict[int, Dict[str, Any]] = {}
                for tmpl_text, g in tmpl_groups.items():
                    lang_row = safe_optional(g.iloc[0].get("Language")).strip() or language_dir
                    ds_id = ensure_template_dataset(project_name, metric_category, lang_row, tmpl_text)
                    tmpl_meta[ds_id] = {
                        "template": (tmpl_text.strip() or NA_STR),
                        "regex": build_template_regex(tmpl_text),
                        "lang": lang_row,
                        "eval_rows": g,
                    }

                # Assign response instances to datasets using regex+score, capped by communities hint
                instances_by_ds: Dict[int, List[Dict[str, str]]] = {ds: [] for ds in tmpl_meta.keys()}
                unassigned: List[Dict[str, str]] = []

                for rr in resp_rows:
                    inst = rr["Instance"]
                    best_ds = None
                    best_score = -1
                    for ds_id, info in tmpl_meta.items():
                        if info["regex"].match(inst):
                            score = 10_000 + score_template_match(info["template"], inst)
                        else:
                            score = score_template_match(info["template"], inst)
                        if score > best_score:
                            best_score = score
                            best_ds = ds_id
                    if best_ds is None or best_score <= 0:
                        unassigned.append(rr)
                        continue
                    if len(instances_by_ds[best_ds]) < max_instances_hint:
                        instances_by_ds[best_ds].append(rr)
                    else:
                        unassigned.append(rr)

                # Emit observation+element for each instance, and attach measures by modulo over eval_rows
                for ds_id, info in tmpl_meta.items():
                    inst_list = instances_by_ds.get(ds_id, [])
                    if not inst_list and unassigned:
                        take = min(len(unassigned), max_instances_hint)
                        inst_list = [unassigned.pop(0) for _ in range(take)]

                    if not inst_list:
                        continue

                    eval_rows = info["eval_rows"]
                    if len(eval_rows) == 0:
                        continue

                    for j, rr in enumerate(inst_list):
                        instance_val = rr.get("Instance") or NA_STR
                        response_text = rr.get("Response") or NA_STR

                        instance_hash = hashlib.sha1(instance_val.encode("utf-8", errors="ignore")).hexdigest()[:10]
                        obs_id = stable_int(
                            f"observation::{project_name}/{metric_category}::{info['lang']}::{ts}::{model_name}::ds{ds_id}::inst{instance_hash}"
                        )

                        desc = response_text.strip()
                        if len(desc) > OBS_DESC_MAXLEN:
                            desc = desc[:OBS_DESC_MAXLEN]

                        rows["observation"].append({
                            "id": obs_id,
                            "observer": safe_required(DEFAULT_OBSERVER)[:100],
                            "whenObserved": when_obs,
                            "tool_id": tool_id,
                            "dataset_id": ds_id,
                            "eval_id": eval_id,
                            "name": safe_required(f"{metric_category}:{model_name}:ds[{ds_id}]")[:100],
                            "description": safe_required(desc),
                        })

                        meas_eid = ensure_element(
                            type_spec="template_instance",
                            name=str(ds_id),
                            description=instance_val,
                            project_id=project_id,
                        )

                        rows["evaluation_element"].append({"ref": meas_eid, "eval": eval_id})
                        rows["evaluates_eval"].append({"evaluates": meas_eid, "evalu": eval_id})

                        # Duplicate measures if instances > eval rows (modulo mapping)
                        er = eval_rows.iloc[j % len(eval_rows)]
                        add_measure(obs_id, meas_eid, "evaluation", er.get("Evaluation"))
                        add_measure(obs_id, meas_eid, "oracle_prediction", er.get("Oracle Prediction"))
                        add_measure(obs_id, meas_eid, "oracle_evaluation", er.get("Oracle Evaluation"))

    table_keys = {
        "project": ["id"],
        "tool": ["id"],
        "configuration": ["id"],
        "confparam": ["id"],
        "datashape": ["id"],
        "element": ["id"],
        "dataset": ["id"],
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