"""Atomic per-sample stage checkpoints for deterministic interruption recovery."""

from __future__ import annotations

import hashlib
import json
import os


class StageCheckpointStore:
    def __init__(self, directory=None, enabled=False, fingerprint="", readonly_sources=()):
        self.directory = os.path.abspath(directory) if directory else None
        self.enabled = bool(enabled and self.directory)
        self.fingerprints = fingerprint if isinstance(fingerprint, dict) else {}
        self.fingerprint = "" if self.fingerprints else str(fingerprint or "")
        self.readonly_sources = tuple(os.path.abspath(path) for path in readonly_sources)
        self.reuse_events = []
        if self.directory:
            os.makedirs(self.directory, exist_ok=True)

    def fingerprint_for(self, stage):
        return self.fingerprints.get(stage, self.fingerprints.get("default", self.fingerprint))

    @staticmethod
    def _key(sample):
        raw = sample.get("raw", {}) or {}
        sample_id = str(raw.get("id") or sample.get("index"))
        digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
        return sample_id, digest

    @staticmethod
    def input_identity(sample):
        raw = sample.get("raw", {}) or {}
        caption = sample.get("caption", raw.get("caption", ""))
        image = sample.get("image")
        image_bytes = raw.get("image_bytes")
        if isinstance(image_bytes, (bytes, bytearray)):
            image_digest = hashlib.sha256(image_bytes).hexdigest()
        elif image is not None and hasattr(image, "tobytes"):
            image_digest = hashlib.sha256(
                str((image.mode, image.size)).encode() + image.tobytes()
            ).hexdigest()
        else:
            image_digest = raw.get("image_sha256", "")
        return hashlib.sha256(json.dumps(
            {"caption": caption, "image": image_digest},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest()

    def _path(self, stage, sample):
        _, digest = self._key(sample)
        safe_stage = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in str(stage)
        )
        return os.path.join(self.directory, f"{safe_stage}_{digest}.json")

    def load(self, stage, sample):
        if self.enabled:
            own = self._load_path(self._path(stage, sample), stage, sample)
            if own is not None:
                return own
        # Only pristine upstream stages may cross experiment directories.
        if stage in {"visual_grounding", "initial_reasoning", "visual_candidate", "text_candidate"}:
            _, digest = self._key(sample)
            for source in self.readonly_sources:
                path = os.path.join(source, f"{stage}_{digest}.json")
                try:
                    payload = self._load_path(path, stage, sample)
                except RuntimeError as error:
                    self.reuse_events.append({"path": path, "reused": False, "reason": str(error)})
                    continue
                if payload is not None:
                    self.reuse_events.append({"path": path, "reused": True})
                    return payload
        return None

    def _load_path(self, path, stage, sample):
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        sample_id, _ = self._key(sample)
        if record.get("sample_id") != sample_id or record.get("stage") != stage:
            raise RuntimeError(f"Stage checkpoint identity mismatch: {path}")
        if str(record.get("fingerprint") or "") != self.fingerprint_for(stage):
            raise RuntimeError(f"Stage checkpoint configuration mismatch: {path}")
        if record.get("input_identity") != self.input_identity(sample):
            raise RuntimeError(f"Stage checkpoint input mismatch: {path}")
        if record.get("payload_sha256") != self.payload_digest(record.get("payload")):
            raise RuntimeError(f"Stage checkpoint payload integrity mismatch: {path}")
        return record.get("payload")

    @staticmethod
    def payload_digest(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def save(self, stage, sample, payload):
        if not self.directory:
            return
        path = self._path(stage, sample)
        sample_id, _ = self._key(sample)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {"schema_version": "4.0", "stage": stage,
                 "payload_sha256": self.payload_digest(payload),
                 "input_identity": self.input_identity(sample),
                 "sample_id": sample_id, "fingerprint": self.fingerprint_for(stage),
                 "payload": payload},
                handle, ensure_ascii=False, sort_keys=True,
            )
        os.replace(temporary, path)
