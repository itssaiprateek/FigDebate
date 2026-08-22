class FigDebate:
    """
    Public phase-neutral API for the FigDebate system.

    The API wraps the canonical engine: visual grounding, claim extraction,
    evidence comparison, arbitration, selective debate, and feedback memory.
    """

    def __init__(self):
        from engine.orchestrator import Orchestrator

        self.orchestrator = Orchestrator()

    def predict(self, image, caption):
        sample = {
            "image": image,
            "caption": caption,
        }

        return self.orchestrator.run_sample(sample)
