import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.paths import PROJECT_ROOT, scenario_runs_dir

SCENARIO_FILES = {
    "train": PROJECT_ROOT / "scenarios" / "training_scenarios.json",
    "inference": PROJECT_ROOT / "scenarios" / "inference_scenarios.json",
}

SCRIPT_BY_PHASE = {
    "train": PROJECT_ROOT / "src" / "train.py",
    "inference": PROJECT_ROOT / "src" / "prompt.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predefined sustainability scenarios.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available scenarios")
    list_parser.add_argument("--phase", choices=["train", "inference", "all"], default="all")

    run_parser = subparsers.add_parser("run", help="Run one scenario")
    run_parser.add_argument("--phase", choices=["train", "inference"], required=True)
    run_parser.add_argument("--scenario-id", required=True)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--checkpoint-path", default=None)
    run_parser.add_argument("--skip-if-complete", action="store_true")
    run_parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    sweep_parser = subparsers.add_parser("sweep", help="Run all scenarios in a phase")
    sweep_parser.add_argument("--phase", choices=["train", "inference"], required=True)
    sweep_parser.add_argument("--run-prefix", default="")
    sweep_parser.add_argument("--checkpoint-path", default=None)
    sweep_parser.add_argument("--skip-if-complete", action="store_true")
    sweep_parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    return parser.parse_args()


def load_phase_scenarios(phase: str) -> dict:
    scenario_file = SCENARIO_FILES[phase]
    with scenario_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("phase") != phase:
        raise ValueError(f"Scenario file {scenario_file} has wrong phase value.")
    return data


def list_scenarios(phase: str) -> None:
    phases = [phase] if phase != "all" else ["train", "inference"]
    for item in phases:
        config = load_phase_scenarios(item)
        print(f"[{item}] {len(config['scenarios'])} scenarios")
        for scenario in config["scenarios"]:
            print(f"  - {scenario['id']}: {scenario.get('description', '')}")


def get_scenario(phase: str, scenario_id: str) -> tuple[dict, dict]:
    config = load_phase_scenarios(phase)
    scenario = next((s for s in config["scenarios"] if s["id"] == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Unknown {phase} scenario id: {scenario_id}")
    return config, scenario


def scenario_is_complete(phase: str, scenario_id: str) -> bool:
    latest_path = scenario_runs_dir(phase, scenario_id) / "latest_run.json"
    if not latest_path.exists():
        return False
    with latest_path.open("r", encoding="utf-8") as f:
        latest = json.load(f)
    return latest.get("status") == "completed"


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_training_checkpoint_from_latest(training_scenario_id: str) -> tuple[str | None, str | None]:
    latest_path = scenario_runs_dir("train", training_scenario_id) / "latest_run.json"
    if not latest_path.exists():
        return None, f"missing latest run metadata: {latest_path}"

    with latest_path.open("r", encoding="utf-8") as f:
        latest = json.load(f)

    if latest.get("status") != "completed":
        return None, f"latest training run is not completed for scenario '{training_scenario_id}'"

    checkpoint_path = latest.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        return None, (
            f"latest run metadata for training scenario '{training_scenario_id}' does not include checkpoint_path"
        )

    resolved_checkpoint = _resolve_path(checkpoint_path)
    if not resolved_checkpoint.exists():
        return None, f"checkpoint referenced in latest run metadata does not exist: {resolved_checkpoint}"

    return str(resolved_checkpoint), str(latest_path)


def _apply_inference_checkpoint(
    *,
    merged: dict,
    phase: str,
    scenario_id: str,
    checkpoint_path: str | None,
) -> tuple[dict | None, str | None]:
    merged_config = dict(merged)
    merged_model_scenario_id = merged_config.pop("model_training_scenario_id", None)

    if checkpoint_path:
        resolved_checkpoint = _resolve_path(checkpoint_path)
        if not resolved_checkpoint.exists():
            return None, f"checkpoint argument does not exist: {resolved_checkpoint}"
        merged_config["checkpoint_path"] = str(resolved_checkpoint)
        return merged_config, None

    if not isinstance(merged_model_scenario_id, str) or not merged_model_scenario_id.strip():
        return None, (
            "inference scenario is missing model_training_scenario_id and no --checkpoint-path override was provided"
        )

    resolved_checkpoint, latest_source = _resolve_training_checkpoint_from_latest(merged_model_scenario_id)
    if resolved_checkpoint is None:
        return None, (
            f"unable to resolve checkpoint from training scenario '{merged_model_scenario_id}': {latest_source}"
        )

    print(
        f"Using checkpoint for {phase}:{scenario_id} from training scenario "
        f"'{merged_model_scenario_id}': {resolved_checkpoint} (source: {latest_source})"
    )
    merged_config["checkpoint_path"] = resolved_checkpoint
    return merged_config, None


def build_command(
    phase: str,
    scenario_id: str,
    run_id: str | None,
    checkpoint_path: str | None,
    extra: list[str],
) -> tuple[list[str] | None, str | None]:
    phase_config, scenario = get_scenario(phase, scenario_id)
    defaults = phase_config.get("defaults", {})
    overrides = scenario.get("overrides", {})

    if phase == "inference":
        default_length = defaults.get("max_new_tokens")
        override_length = overrides.get("max_new_tokens", default_length)
        if override_length != default_length:
            raise ValueError(
                "Inference comparability guard: max_new_tokens must remain fixed across predefined scenarios."
            )

    merged = {**defaults, **overrides}
    if phase == "inference":
        merged, skip_reason = _apply_inference_checkpoint(
            merged=merged,
            phase=phase,
            scenario_id=scenario_id,
            checkpoint_path=checkpoint_path,
        )
        if merged is None:
            return None, skip_reason

    cmd = [sys.executable, str(SCRIPT_BY_PHASE[phase]), "--scenario-id", scenario_id]
    if run_id:
        cmd.extend(["--run-id", run_id])

    for key, value in merged.items():
        flag = f"--{key.replace('_', '-')}"
        cmd.extend([flag, str(value)])

    if extra:
        cmd.extend(extra)

    return cmd, None


def run_one(
    phase: str,
    scenario_id: str,
    run_id: str | None,
    checkpoint_path: str | None,
    skip_if_complete: bool,
    extra: list[str],
) -> int:
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if skip_if_complete and scenario_is_complete(phase, scenario_id):
        print(f"Skipping {phase}:{scenario_id} (already completed).")
        return 0

    cmd, skip_reason = build_command(phase, scenario_id, resolved_run_id, checkpoint_path, extra)
    if cmd is None:
        print(f"Warning: Skipping {phase}:{scenario_id} ({skip_reason})")
        return 0

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)

    run_dir = scenario_runs_dir(phase, scenario_id) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "runner.log").open("w", encoding="utf-8") as f:
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            f.write("\n[stderr]\n")
            f.write(result.stderr)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode


def run_sweep(
    phase: str,
    run_prefix: str,
    checkpoint_path: str | None,
    skip_if_complete: bool,
    extra: list[str],
) -> int:
    config = load_phase_scenarios(phase)
    failures: list[str] = []

    for index, scenario in enumerate(config["scenarios"]):
        scenario_id = scenario["id"]
        run_id = f"{run_prefix}{index:02d}_{scenario_id}" if run_prefix else None
        code = run_one(
            phase=phase,
            scenario_id=scenario_id,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            skip_if_complete=skip_if_complete,
            extra=extra,
        )
        if code != 0:
            failures.append(scenario_id)

    if failures:
        print("Sweep completed with failures:", ", ".join(failures))
        return 1

    print("Sweep completed successfully.")
    return 0


def main():
    args = parse_args()

    if args.command == "list":
        list_scenarios(args.phase)
        return

    if args.command == "run":
        raise SystemExit(
            run_one(
                phase=args.phase,
                scenario_id=args.scenario_id,
                run_id=args.run_id,
                checkpoint_path=args.checkpoint_path,
                skip_if_complete=args.skip_if_complete,
                extra=args.extra,
            )
        )

    if args.command == "sweep":
        raise SystemExit(
            run_sweep(
                phase=args.phase,
                run_prefix=args.run_prefix,
                checkpoint_path=args.checkpoint_path,
                skip_if_complete=args.skip_if_complete,
                extra=args.extra,
            )
        )


if __name__ == "__main__":
    main()
