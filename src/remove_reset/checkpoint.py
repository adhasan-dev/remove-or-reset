"""Immutable safetensors + tagged JSON optimizer/RNG checkpoints; never torch.load.

Only optimizer boundaries (after step/scheduler/zero_grad) are resumable.
An interrupted microbatch is discarded; latest.json points only to a sealed
checkpoint. File fsync + same-volume rename protects process-crash consistency,
not arbitrary power loss, especially on Windows.
"""

from __future__ import annotations

import math
import os
import random
import time
import uuid
from pathlib import Path

from .artifacts import (
    atomic_json,
    read_json,
    seal,
    sha256,
    sync_directory,
    value_hash,
    verify_bundle,
    write_json,
)
from .model import finite_tensor


def pack(value, tensors):
    import torch

    if isinstance(value, torch.Tensor):
        finite_tensor(value, "checkpoint payload")
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"tensor": key}
    if isinstance(value, dict):
        return {"dict": [[pack(k, tensors), pack(v, tensors)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [pack(v, tensors) for v in value]}
    if isinstance(value, list):
        return {"list": [pack(v, tensors) for v in value]}
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Nonfinite checkpoint scalar")
        return {"scalar": value}
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def unpack(value, tensors):
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("Malformed tagged checkpoint JSON")
    key, payload = next(iter(value.items()))
    if key == "tensor":
        if payload not in tensors:
            raise ValueError("Missing checkpoint tensor reference")
        return tensors[payload]
    if key == "dict":
        pairs = [(unpack(k, tensors), unpack(v, tensors)) for k, v in payload]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate checkpoint dictionary keys")
        return result
    if key == "tuple":
        return tuple(unpack(v, tensors) for v in payload)
    if key == "list":
        return [unpack(v, tensors) for v in payload]
    if key == "scalar" and (payload is None or type(payload) in (str, bool, int, float)):
        return payload
    raise ValueError("Unsupported checkpoint JSON tag/value")


def capture_rng():
    import numpy as np
    import torch

    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), *state[2:]),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    name, keys, position, gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_rng(state):
    """Validate safe RNG values using private generators, without mutating global RNG."""
    import numpy as np
    import torch

    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Invalid RNG state fields")
    random.Random().setstate(state["python"])
    name, keys, position, gauss, cached = state["numpy"]
    np.random.RandomState().set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, gauss, cached)
    )
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG device count differs")
        for index, value in enumerate(state["torch_cuda"]):
            torch.Generator(device=f"cuda:{index}").set_state(value)


def named_optimizer(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups, states = [], {}
    for group in optimizer.param_groups:
        item = {key: value for key, value in group.items() if key != "params"}
        item["params"] = [names[id(parameter)] for parameter in group["params"]]
        groups.append(item)
        for parameter in group["params"]:
            states[names[id(parameter)]] = optimizer.state.get(parameter, {})
    return {"groups": groups, "states": states}


def save_checkpoint(run, adapter, optimizer, scheduler, state, identity):
    from safetensors.torch import save_file

    model = adapter.network
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Checkpoint requires a cleared-gradient optimizer boundary")
    root = Path(run) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['updates']:08d}-{uuid.uuid4().hex[:12]}"
    stage, final = root / f".{name}.partial", root / name
    stage.mkdir(exist_ok=False)
    weights = {
        f"parameter::{n}": p.detach().cpu().contiguous().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    weights.update(
        {f"buffer::{n}": b.detach().cpu().contiguous().clone() for n, b in model.named_buffers()}
    )
    for key, tensor in weights.items():
        finite_tensor(tensor, key)
    save_file(weights, str(stage / "adapter.safetensors"))
    tensors = {}
    payload = pack(
        {
            "optimizer": named_optimizer(model, optimizer),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng(),
            "state": state,
        },
        tensors,
    )
    save_file(tensors, str(stage / "training.safetensors"))
    write_json(
        stage / "state.json",
        {"format": "remove-reset-training-checkpoint-v1", "identity": identity, "payload": payload},
    )
    for path in stage.iterdir():
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
    seal(stage, metadata={"identity_sha256": value_hash(identity)})
    verify_bundle(stage)
    os.rename(stage, final)
    sync_directory(root)
    atomic_json(
        Path(run) / "latest.json",
        {
            "checkpoint": name,
            "manifest_sha256": sha256(final / "manifest.json"),
            "updates": state["updates"],
        },
    )
    return final


def resolve_checkpoint(run, name="latest"):
    from .artifacts import local_path

    root = Path(run) / "checkpoints"
    pointer = read_json(Path(run) / "latest.json") if name == "latest" else None
    name = pointer["checkpoint"] if pointer is not None else name
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise ValueError("Checkpoint must be an immutable leaf directory name")
    result = local_path(root, name)
    if pointer is not None and sha256(result / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("Latest checkpoint pointer checksum differs")
    return result


def read_checkpoint(path, identity):
    from safetensors.torch import load_file

    manifest = verify_bundle(path)
    if set(manifest["files"]) != {
        "adapter.safetensors",
        "training.safetensors",
        "state.json",
    } or manifest["identity_sha256"] != value_hash(identity):
        raise ValueError("Checkpoint file set or identity differs")
    doc = read_json(Path(path) / "state.json")
    if (
        doc.get("format") != "remove-reset-training-checkpoint-v1"
        or doc.get("identity") != identity
    ):
        raise ValueError("Checkpoint resume identity differs")
    weights = load_file(str(Path(path) / "adapter.safetensors"), device="cpu")
    tensors = load_file(str(Path(path) / "training.safetensors"), device="cpu")
    for name, tensor in {**weights, **tensors}.items():
        finite_tensor(tensor, name)
    payload = unpack(doc["payload"], tensors)
    return weights, payload


def validate_weights(model, weights):
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    expected = {f"parameter::{n}" for n, p in parameters.items() if p.requires_grad}
    expected |= {f"buffer::{n}" for n in buffers}
    if set(weights) != expected:
        raise ValueError("Checkpoint trainable parameter/buffer names differ")
    targets = {}
    for key, tensor in weights.items():
        kind, name = key.split("::", 1)
        target = parameters[name] if kind == "parameter" else buffers[name]
        if target.shape != tensor.shape or target.dtype != tensor.dtype:
            raise ValueError(f"Checkpoint tensor shape/dtype differs: {key}")
        targets[key] = target
    return targets


def load_checkpoint(
    path, adapter, optimizer, scheduler, identity, *, expected_order=None, records=None, recipe=None
):
    """Validate identities, names, shapes, moments, sampler and scheduler before edits."""
    import torch

    weights, payload = read_checkpoint(path, identity)
    model = adapter.network
    targets = validate_weights(model, weights)
    current, saved = named_optimizer(model, optimizer), payload["optimizer"]
    names = [name for group in current["groups"] for name in group["params"]]
    if [g["params"] for g in saved["groups"]] != [g["params"] for g in current["groups"]] or set(
        saved["states"]
    ) != set(names):
        raise ValueError("Optimizer named parameter groups differ")
    parameters = dict(model.named_parameters())
    updates = payload["state"]["updates"]
    if type(updates) is not int or updates < 0:
        raise ValueError("Invalid completed update count")
    for name, moments in saved["states"].items():
        if not moments:
            if updates:
                raise ValueError("Missing Adam state after completed updates")
            continue
        if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unsupported Adam state")
        for key in ("exp_avg", "exp_avg_sq"):
            if (
                moments[key].shape != parameters[name].shape
                or moments[key].dtype != parameters[name].dtype
            ):
                raise ValueError("Adam moment shape/dtype differs")
        if moments["step"].numel() != 1 or moments["step"].item() != updates:
            raise ValueError("Adam step count differs from completed updates")
    if payload["scheduler"]["last_epoch"] != updates:
        raise ValueError("Scheduler step count differs")
    order, cursor = payload["state"]["order"], payload["state"]["cursor"]
    if (
        not isinstance(order, list)
        or any(type(i) is not int for i in order)
        or sorted(order) != list(range(len(order)))
        or type(cursor) is not int
        or not 0 <= cursor <= len(order)
    ):
        raise ValueError("Checkpoint sampler permutation/cursor invalid")
    if expected_order is not None and order != expected_order:
        raise ValueError("Checkpoint full training order differs")
    if records is not None and recipe is not None:
        state = payload["state"]
        expected_updates = math.ceil(cursor / recipe.effective_batch_size)
        if (
            len(order) != len(records)
            or state["examples"] != cursor
            or updates != expected_updates
            or (cursor != len(records) and cursor % recipe.effective_batch_size)
            or type(state["complete"]) is not bool
            or state["complete"] != (cursor == len(records))
            or state["nonpadding_tokens"] != sum(len(records[i].input_ids) for i in order[:cursor])
            or state["target_tokens"] != sum(records[i].target_tokens for i in order[:cursor])
        ):
            raise ValueError("Checkpoint completed sampler/token counters differ")
    for saved_group, current_group in zip(saved["groups"], current["groups"], strict=True):
        if set(saved_group) != set(current_group):
            raise ValueError("Optimizer hyperparameter fields differ")
        if any(
            saved_group[key] != current_group[key]
            for key in current_group
            if key not in ("lr", "params")
        ):
            raise ValueError("Optimizer fixed hyperparameters differ")
    if (
        payload["scheduler"]["base_lrs"] != scheduler.base_lrs
        or payload["scheduler"]["_step_count"] != updates + 1
        or payload["scheduler"]["_last_lr"] != [g["lr"] for g in saved["groups"]]
    ):
        raise ValueError("Saved scheduler/optimizer learning-rate state differs")
    expected_lrs = [
        base * factor(updates)
        for base, factor in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)
    ]
    if expected_lrs != [group["lr"] for group in saved["groups"]]:
        raise ValueError("Saved learning rate does not follow the fixed original schedule")
    validate_rng(payload["rng"])
    groups, states, offset = [], {}, 0
    for group in saved["groups"]:
        indices = list(range(offset, offset + len(group["params"])))
        states.update(
            {i: saved["states"][n] for i, n in zip(indices, group["params"], strict=True)}
        )
        groups.append({**group, "params": indices})
        offset += len(indices)
    with torch.no_grad():
        for key, target in targets.items():
            target.copy_(weights[key].to(device=target.device))
    # These methods consume only the already validated plain in-memory state.
    optimizer.load_state_dict({"state": states, "param_groups": groups})
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])  # LAST: reconstruction must not consume restored RNG.
    return payload["state"]


def load_final_adapter(path, adapter, identity):
    import torch

    weights, payload = read_checkpoint(path, identity)
    state = payload["state"]
    if not state["complete"] or state["cursor"] != len(state["order"]):
        raise ValueError("Evaluation requires the completed epoch, not a selected checkpoint")
    targets = validate_weights(adapter.network, weights)
    with torch.no_grad():
        for key, target in targets.items():
            target.copy_(weights[key].to(device=target.device))
    adapter.freeze()
    return state


class CheckpointClock:
    """Cooperative boundary deadlines, not a hard-realtime guarantee."""

    def __init__(self, recipe):
        self.recipe = recipe
        self.last_time, self.last_update = time.monotonic(), 0
        self.save_seconds, self.max_gap, self.missed = 0.0, 0.0, False
        self.requested = False

    def due(self, update, predicted_seconds=0.0):
        age = time.monotonic() - self.last_time
        self.missed |= age > self.recipe.checkpoint_max_gap_seconds
        self.requested |= (
            update - self.last_update >= self.recipe.checkpoint_updates
            or age + predicted_seconds + self.save_seconds >= self.recipe.checkpoint_seconds
        )
        return self.requested

    def saved(self, update, duration):
        self.max_gap = max(self.max_gap, time.monotonic() - self.last_time)
        self.missed |= self.max_gap > self.recipe.checkpoint_max_gap_seconds
        self.last_time, self.last_update = time.monotonic(), update
        self.save_seconds = max(self.save_seconds, duration)
        self.requested = False
