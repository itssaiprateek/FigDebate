# Setup verification — 2026-09-28

Verified on Windows x86-64, Python 3.11.16, PyTorch 2.5.1+cu121 and an
NVIDIA RTX 4060 Laptop GPU. Linux dependency markers are locked, but Linux
execution was not tested in this session.

- Created an independent project-local Python environment and installed the lock.
- Confirmed NumPy/PyTorch imports do not reference the old checkout or research runtime.
- Ran the documented bootstrap end to end: READY, pip check passed, 693 software
  tests completed with one expected failure and no unexpected failures.
- Verified all four pinned model sources, tokenizer/config loading, auxiliary
  weight headers, Qwen revision metadata and actual CUDA tensor execution.
- Rebuilt all dataset files from the pinned original parquet shards in an empty
  directory, then verified 3,992 train, 573 validation and 569 test rows plus the
  50-case training subset against their sources.
- Installed the verified native-byte preparation locally. Historical processed
  files and their provenance remain in `.cache/pre-portable-data` and
  `.cache/pre-portable-provenance`; old run outputs were not changed.

Large assets were copied from existing pinned caches after downloads stalled;
Hub metadata requests and authentication were exercised. A completely uncached
multi-gigabyte download was not completed in this validation session.
No full inference or semantic accuracy evaluation was run as part of setup.
