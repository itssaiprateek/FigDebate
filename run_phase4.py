"""Deprecated compatibility entry point for the official FigDebate runner."""

import warnings

from run_figdebate import main


if __name__ == "__main__":
    warnings.warn(
        "run_phase4.py is deprecated; use run_figdebate.py.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
