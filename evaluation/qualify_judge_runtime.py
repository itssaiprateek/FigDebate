"""Isolated current-runtime smoke check; never an accuracy experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from dataset.loaders import load_split, decode_image
from models.judge_model import QwenJudgeModel
from engine.output_contracts import object_schema, TEXT, validate_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--hardware-profile", default="paper-8gb")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    raw = load_split("vflute_train_dev50")[0]
    schema = object_schema({"observation": TEXT})
    started = time.perf_counter()
    record = {"purpose": "isolated_runtime_qualification_not_accuracy",
              "sample_id": raw["id"], "image_sha256": raw["image_sha256"],
              "gold_visible": False, "hardware_profile": args.hardware_profile,
              "source_sha256": hashlib.sha256(Path("models/judge_model.py").read_bytes()).hexdigest()}
    try:
        runtime = QwenJudgeModel(hardware_profile=args.hardware_profile)
        answer, seconds = runtime.generate(decode_image(raw["image_bytes"]),
            'Inspect the image. Return JSON only: {"observation":"one short visible fact"}.',
            max_new_tokens=80, json_schema=schema)
        record.update(answer=answer, seconds=seconds, diagnostics=runtime._last_generation_diagnostics,
                      schema_valid=validate_shape(json.loads(answer), schema), execution_status="SUCCEEDED")
    except Exception as error:
        record.update(error=str(error), error_type=type(error).__name__, execution_status="FAILED")
    record["wall_seconds"] = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    if record["execution_status"] != "SUCCEEDED" or not record.get("schema_valid"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
