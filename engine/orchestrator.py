"""Public single-sample wrapper around the canonical stagewise engine."""

from engine.batch_runner import StagewiseRunner


class Orchestrator:
    """Keep the public API on exactly the same execution path as experiments."""

    def __init__(
        self,
        *,
        feedback_mode="disabled",
        debate_mode="enabled",
        evidence_mode="enabled",
        judge_mode="disabled",
        judge_scope="escalated",
        verified_feedback_path=None,
    ):
        print("=" * 70)
        print("FigDebate Engine Ready")
        print("=" * 70)
        self.runner = StagewiseRunner(
            feedback_mode=feedback_mode,
            verified_feedback_path=verified_feedback_path,
            debate_mode=debate_mode,
            evidence_mode=evidence_mode,
            judge_mode=judge_mode,
            judge_scope=judge_scope,
        )
        # Preserve the previously public engine handle for callers that inspect it.
        self.debate = self.runner.debate

    def run_sample(self, sample):
        """Run one sample without maintaining a second, divergent pipeline."""
        if "image" not in sample or "caption" not in sample:
            raise ValueError("A sample requires both 'image' and 'caption'.")
        raw = dict(sample)
        raw.setdefault("id", "api_sample")
        normalized = {
            "index": 0,
            "image": sample["image"],
            "caption": sample["caption"],
            "raw": raw,
        }
        captured = {}

        def receive(_index, _raw, result, _elapsed):
            captured["result"] = result

        self.runner.run_samples([normalized], receive)
        return captured["result"]
