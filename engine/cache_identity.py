"""Conservative stage-specific source/config identities, including dirty files."""
import hashlib
import ast
import json
from pathlib import Path


def local_dependencies(root, seeds):
    """Static local import closure, including dirty/untracked imported helpers."""
    root = Path(root).resolve()
    pending, found = list(seeds), set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(relative)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    parents = Path(relative).parent.parts
                    prefix = ".".join((*parents[:len(parents) - node.level + 1], prefix)).strip(".")
                names = [prefix] + [prefix + "." + item.name for item in node.names]
            for name in names:
                parts = name.split(".")
                candidates = ["/".join(parts) + ".py"]
                candidates += ["/".join(parts[:i]) + "/__init__.py" for i in range(1, len(parts) + 1)]
                pending.extend(item for item in candidates if (root / item).is_file() and item not in found)
    return sorted(found)


def stage_fingerprints(root, config):
    root = Path(root)
    def digest(paths, settings):
        files = {}
        for relative in local_dependencies(root, paths + ["engine/cache_identity.py"]):
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashlib.sha256(json.dumps({"files": files, "settings": settings},
                                         sort_keys=True).encode()).hexdigest()
    common = {key: config.get(key) for key in ("seed", "hardware_profile", "runtime_environment", "deterministic_environment")}
    visual_paths = ["agents/visual_grounding.py", "agents/visual_adapter.py", "models/vision_model.py",
                    "engine/runtime_profile.py", "engine/gpu_manager.py", "engine/reproducibility.py"]
    # Full downstream fingerprint is intentionally conservative; upstream vision
    # can still be reused when only judge, calibration or reporting code changes.
    visual = digest(visual_paths, dict(common, model=config.get("model_vision_revision")))
    base = hashlib.sha256(json.dumps(dict(common,
        pipeline=config.get("pipeline_source_sha256"),
        evidence=config.get("evidence_mode"), feedback=config.get("feedback_mode"),
        feedback_file=config.get("verified_feedback_sha256")), sort_keys=True).encode()).hexdigest()
    full = hashlib.sha256(json.dumps(dict(common, base=base,
        ablation=config.get("ablation_signature")), sort_keys=True).encode()).hexdigest()
    return {"visual_grounding": visual, "initial_reasoning": base,
            "visual_candidate": digest(visual_paths + ["engine/candidate_cases.py",
                "agents/multimodal_judge.py"], dict(common, stage="visual_candidate")),
            "text_candidate": base, "default": full}
