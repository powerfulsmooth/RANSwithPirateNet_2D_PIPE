"""Misc helpers (mirrors jaxpi/utils.py): checkpoint save/load."""
import os
from flax import serialization


def save_params(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))


def load_params(params_like, path):
    with open(path, "rb") as f:
        return serialization.from_bytes(params_like, f.read())
