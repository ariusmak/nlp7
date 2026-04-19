from __future__ import annotations

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_yaml(name: str) -> dict:
    with open(CONFIGS_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else REPO_ROOT / p


def paths() -> dict:
    cfg = load_yaml("paths.yaml")

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, str):
            return resolve(node)
        return node

    return walk(cfg)


def universe() -> dict:
    return load_yaml("universe.yaml")


def sentiment_config() -> dict:
    return load_yaml("sentiment.yaml")
