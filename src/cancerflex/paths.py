"""Configurable locations used throughout CancerFlex.

Every path that was fixed in the research code is read from an environment
variable here, so the same code runs on any machine:

* ``CANCERFLEX_ROOT``       – workspace root holding ``Experiment/datasets/...``
                              (training shards, per-pillar manifests, caches) and
                              ``eval_results/``. Defaults to the current directory.
* ``CANCERFLEX_BASE_MODEL`` – local path or Hub id of the MedGemma 1.5 4B base
                              checkpoint (processor + initial weights).
* ``CANCERFLEX_GEMMA3N_BASE`` – kept only for symbol compatibility; the Gemma-3n
                              reference arm is not part of this release.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CANCERFLEX_ROOT", os.getcwd())).resolve()
MEDGEMMA_BASE = os.environ.get("CANCERFLEX_BASE_MODEL", "google/medgemma-1.5-4b-it")
GEMMA3N_BASE = os.environ.get("CANCERFLEX_GEMMA3N_BASE", "google/gemma-3n-E4B")
