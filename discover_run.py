from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Optional


RUN_TS_RE = re.compile(r"^\d{8}_\d{6}$")  # 20251111_142209
GLOBAL_RE = re.compile(r".*_global_evaluation\.csv$")
EVALS_RE  = re.compile(r".*_evaluations\.csv$")
RESP_RE   = re.compile(r".*_responses\.csv$")


@dataclass
class RunPaths:
    global_csv: str
    evals_csv: str
    responses_csv: str
    config_json: str


@dataclass
class RunInfo:
    run_id: str
    timestamp_dir: str
    test_name: Optional[str]
    provider_family: Optional[str]
    language: Optional[str]
    model: Optional[str]
    paths: RunPaths


def find_one(folder: Path, pattern: re.Pattern) -> Optional[Path]:
    hits = [p for p in folder.glob("*.csv") if pattern.match(p.name)]
    return hits[0] if hits else None


def discover_runs(data_root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []

    # Look for directories whose name matches a timestamp and which contain language subfolders etc.
    for ts_dir in data_root.rglob("*"):
        if not ts_dir.is_dir():
            continue
        if not RUN_TS_RE.match(ts_dir.name):
            continue

        # In your structure: ts_dir/<lang>/*.csv and config.json
        for lang_dir in ts_dir.iterdir():
            if not lang_dir.is_dir():
                continue

            config = lang_dir / "config.json"
            global_csv = find_one(lang_dir, GLOBAL_RE)
            evals_csv  = find_one(lang_dir, EVALS_RE)
            resp_csv   = find_one(lang_dir, RESP_RE)

            if not (config.exists() and global_csv and evals_csv and resp_csv):
                continue

            # Infer metadata from path pieces when possible
            parts = lang_dir.parts
            language = lang_dir.name

            model = None
            # If you later add a model directory (like .../en_us/OpenAIGPT35Turbo/*.csv)
            # this will still work because we can detect parent folder name.
            # For now, infer model from CSV contents later in accumulator.
            # But keep a placeholder.
            model = None

            # Try to infer test_name/provider_family from path
            # e.g. data/mistral/test-ageism/20251111_142209/en_us/...
            provider_family = None
            test_name = None
            for i, p in enumerate(parts):
                if p == "data":
                    # next might be "experiments" or "mistral"
                    if i + 1 < len(parts):
                        provider_family = parts[i + 1]
                if p.startswith("test-"):
                    test_name = p

            run_id = "/".join([x for x in [provider_family, test_name, ts_dir.name, language] if x])

            runs.append(
                RunInfo(
                    run_id=run_id,
                    timestamp_dir=ts_dir.name,
                    test_name=test_name,
                    provider_family=provider_family,
                    language=language,
                    model=model,
                    paths=RunPaths(
                        global_csv=str(global_csv),
                        evals_csv=str(evals_csv),
                        responses_csv=str(resp_csv),
                        config_json=str(config),
                    ),
                )
            )

    # Stable ordering
    runs.sort(key=lambda r: (r.provider_family or "", r.test_name or "", r.timestamp_dir, r.language or ""))
    return runs


if __name__ == "__main__":
    data_root = Path("data")
    runs = discover_runs(data_root)
    out = {"data_root": str(data_root), "n_runs": len(runs), "runs": [asdict(r) for r in runs]}

    manifest_path = data_root / "data_accumulated" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote manifest with {len(runs)} runs to: {manifest_path}")
