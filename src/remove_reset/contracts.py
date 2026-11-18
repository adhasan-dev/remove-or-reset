"""Small research-bench contracts, adapted for this standalone source package.

The ToolSpec/ModelAdapter boundary follows the source bench: pure Python tools
declare parameters and return metrics/tables/figures/notes/arrays. Nothing here
imports the source bench, a UI, or a transport. Heavy libraries are lazy imports.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class ModelInfo:
    name: str
    n_layers: int
    d_model: int
    d_mlp: int
    vocab_size: int
    max_seq_len: int
    n_parameters: int
    device: str
    fingerprint: str
    revision: str
    extra: dict = field(default_factory=dict)


@dataclass
class Trace:
    """One prompt: CPU logits [token,vocabulary]; no silently guessed activations."""

    tokens: list[str]
    token_ids: list[int]
    logits: Any
    activations: dict[str, Any]


class ModelAdapter(Protocol):
    info: ModelInfo

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def trace(self, prompt: str, interventions: Sequence = ()) -> Trace: ...
    def project(self, residual: Any) -> Any: ...


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    default: Any
    choices: tuple[str, ...] = ()
    help: str = ""

    def validate(self, value):
        if self.kind in ("text", "choice"):
            if not isinstance(value, str):
                raise ValueError(f"{self.name} must be text")
            if self.kind == "choice" and value not in self.choices:
                raise ValueError(f"Unsupported {self.name}: {value!r}")
        elif self.kind == "integer":
            if type(value) is not int:
                raise ValueError(f"{self.name} must be an integer")
        elif self.kind == "number":
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{self.name} must be finite")
        elif self.kind == "boolean":
            if type(value) is not bool:
                raise ValueError(f"{self.name} must be boolean")
        else:
            raise ValueError("Unsupported parameter type")
        return value


@dataclass
class ToolContext:
    model: ModelAdapter | None
    seed: int
    run_dir: Path


@dataclass
class ToolResult:
    title: str
    summary: str
    metrics: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    figures: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    arrays: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    id: str
    title: str
    description: str
    parameters: tuple[Param, ...]
    run: Callable[[ToolContext, dict], ToolResult]
    requires_model: bool = False

    def validate(self, parameters):
        unknown = set(parameters) - {p.name for p in self.parameters}
        if unknown:
            raise ValueError(f"Unknown parameters for {self.id}: {sorted(unknown)}")
        return {p.name: p.validate(parameters.get(p.name, p.default)) for p in self.parameters}

    def describe(self):
        return {key: value for key, value in asdict(self).items() if key != "run"}
