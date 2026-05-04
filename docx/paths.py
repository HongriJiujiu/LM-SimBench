# -*- coding: utf-8 -*-
"""
Portable path resolution for plotting scripts — no absolute machine-local paths.

The **repository root** is the directory that contains ``benchmark/`` and (typically)
the unpacked model CSV folders such as ``qwen3_0.6b_4k/``.

Environment overrides (optional):

``LM_SIMBENCH_ROOT``
    Repository root inferred from filesystem when unset.

``LM_SIMBENCH_DATA_ROOT``
    Root that contains ``<model_name>/…`` CSV folders. Defaults to ``repo_root()``.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """
    Repository root (parent of ``benchmark/``).

    Set ``LM_SIMBENCH_ROOT`` if this layout differs on your machine.
    """
    env = os.environ.get("LM_SIMBENCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def dataset_root() -> Path:
    """Root containing per-model CSV directories (defaults to ``repo_root()``)."""
    env = os.environ.get("LM_SIMBENCH_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root()


def figures_case1_dir() -> Path:
    p = repo_root() / "case_1" / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_case2_dir() -> Path:
    p = repo_root() / "case_2" / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_case2_rank_breakdown_dir() -> Path:
    p = repo_root() / "case_2" / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p
