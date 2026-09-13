"""Deterministic hardware profiles for bounded multimodal inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    minimum_total_vram_gb: float
    agent1_max_pixels: int
    judge_max_pixels: int
    agent1_default_tokens: int
    agent1_ocr_tokens: int
    judge_output_tokens: int
    judge_context_items: int
    oom_retry_pixels: tuple[int, ...]
    judge_text_tokens: int = 3072
    judge_total_tokens: int = 6144
    judge_max_seconds: float = 120.0
    tribunal_protocol: str = "evidence-review-4.0"
    judge_case_seconds: float = 240.0
    judge_retrieval_cycles: int = 2
    judge_evidence_first: bool = True

    def as_dict(self):
        payload = asdict(self)
        payload["oom_retry_pixels"] = list(self.oom_retry_pixels)
        return payload


PROFILES = {
    # Canonical cross-machine evaluation contract.  It intentionally matches
    # the conservative 8 GB budget so a larger GPU changes speed, not inputs.
    "paper-8gb": RuntimeProfile(
        "paper-8gb", 0.0, 2_359_296, 1_048_576, 96, 180, 1024, 18,
        (),
    ),
    "8gb": RuntimeProfile(
        "8gb", 0.0, 2_359_296, 1_048_576, 96, 180, 1024, 18,
        (1_048_576, 589_824, 262_144),
    ),
    "12gb": RuntimeProfile(
        "12gb", 10.0, 3_500_000, 1_572_864, 128, 240, 1024, 22,
        (2_359_296, 1_048_576, 589_824),
    ),
    "16gb": RuntimeProfile(
        "16gb", 14.0, 4_718_592, 2_359_296, 160, 320, 1024, 26,
        (3_500_000, 2_359_296, 1_048_576),
    ),
}


def detected_vram_gb():
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 3)
    except (ImportError, RuntimeError):
        pass
    return None


def resolve_runtime_profile(name="auto", total_vram_gb=None):
    """Resolve an explicit or conservative auto profile without hidden tuning."""
    normalized = str(name or "auto").strip().casefold()
    if normalized in PROFILES:
        return PROFILES[normalized]
    if normalized != "auto":
        raise ValueError(f"Unknown hardware profile: {name}")
    measured = detected_vram_gb() if total_vram_gb is None else float(total_vram_gb)
    if measured is not None and measured >= PROFILES["16gb"].minimum_total_vram_gb:
        return PROFILES["16gb"]
    if measured is not None and measured >= PROFILES["12gb"].minimum_total_vram_gb:
        return PROFILES["12gb"]
    return PROFILES["8gb"]
