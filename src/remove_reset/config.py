"""Fixed scientific recipe; machine paths and pause ceilings are NOT recipe fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

MODEL_ID = "google/gemma-3-4b-pt"
REVISION = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
ARMS = (
    "ordinary_lora",
    "random_removal",
    "random_reset",
    "taylor_removal",
    "taylor_reset",
    "guided_removal",
    "guided_reset",
    "quota_free_guided_removal",
    "quota_free_guided_reset",
)
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
QUOTA_FREE_COUNTS = (
    1588,
    3616,
    2318,
    2440,
    88,
    39,
    39,
    1,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    4,
    16,
    30,
    35,
    58,
    77,
    192,
    428,
    1141,
    2725,
    3072,
    4986,
    6308,
    6876,
    7905,
    7079,
    7060,
    5875,
    5634,
)
TRAIN_SHA256 = "69ab28152f80a7cdf14271c3532fa484a20470001c7f2aadec17cd28f4b1b707"
ORDER_SHA256 = "0fdde173e56e111470582fb628a9fadbece4010748a67a5bf7d5d27d290edbec"
SQL_ID_SHA256 = "f3cfbdb8d0348a40c8ab885e9fb15e5cf4513754d557d363e05074cf4afff035"
VALIDATION_ID_SHA256 = "e57851e537b336e3cb879e925a1896a33f727b4e2d1c1a3b61afa7a328f91fbf"
REASONING_SHA256 = "5fa7d8876dc64ab7e28fe8dbfbe1c657a92a9da8016e01f15c41463deebed278"


def arm_mask(arm: str) -> str | None:
    if arm not in (*ARMS, "base"):
        raise ValueError(f"Unknown arm: {arm!r}")
    if arm in ("base", "ordinary_lora"):
        return None
    if arm.startswith("quota_free_guided_"):
        return "quota_free_guided"
    return arm.split("_", 1)[0]


@dataclass(frozen=True)
class Recipe:
    model_id: str = MODEL_ID
    revision: str = REVISION
    seed: int = 42
    epochs: int = 1
    max_length: int = 2048
    native_dtype: str = "bfloat16"
    adapter_dtype: str = "float32"
    attn_implementation: str = "sdpa"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = PROJECTIONS
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    microbatch: int = 2
    gradient_accumulation: int = 8
    effective_batch_size: int = 16
    warmup_fraction: float = 0.03
    gradient_checkpointing: bool = True
    fused_adamw: bool = True
    deterministic: bool = True
    checkpoint_updates: int = 200
    checkpoint_seconds: int = 1200
    checkpoint_max_gap_seconds: int = 1800

    def document(self) -> dict:
        value = asdict(self)
        value["target_modules"] = list(self.target_modules)
        return value

    @classmethod
    def read(cls, path: str | Path | None = None) -> Recipe:
        from .artifacts import read_json

        return cls.from_document(cls().document() if path is None else read_json(path))

    @classmethod
    def from_document(cls, value: dict) -> Recipe:
        """Allow explicit seed replication, rejecting all other recipe drift.

        This CLI reproduces one fixed study, not arbitrary fine-tuning settings.
        Low-level functions accept a separate tiny test recipe; production never
        substitutes that fixture when CUDA, weights, memory, or data are missing.
        """
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Recipe must contain exactly the documented study fields")
        expected = cls().document()
        for key, reference in expected.items():
            candidate = value[key]
            if key == "seed":
                if type(candidate) is not int or not 0 <= candidate < 2**32:
                    raise ValueError("Training seed must be a nonnegative 32-bit integer")
                continue
            valid_type = (
                type(candidate) is int
                if type(reference) is int
                else type(candidate) in (float, int)
                if type(reference) is float
                else type(candidate) is type(reference)
            )
            if not valid_type or candidate != reference:
                raise ValueError(f"Fixed study recipe differs at {key}")
        return cls(seed=value["seed"])
