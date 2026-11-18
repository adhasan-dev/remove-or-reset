"""Bounded real CPU integration fixture, explicitly NOT the 4B experiment.

Runs the same intervention/safe-loader/PEFT/optimizer/checkpoint/generation code
on a randomly initialized 3-layer FP32 Gemma. No base acquisition, GPU, historical
checkpoint, network, or external dataset is used. Numerical parity here is only
fixture parity, never historical 4B output/performance or bitwise reproduction.
"""

from __future__ import annotations

import gzip
import hashlib
from dataclasses import replace

from .artifacts import (
    canonical_bytes,
    new_output,
    save_result,
    seal,
    sha256,
    value_hash,
    write_json,
)
from .config import ARMS, Recipe
from .contracts import ToolResult
from .data import DataRoot, encode_record, training_order
from .interventions import (
    MASK_FORMAT,
    random_mask,
    remove_channels,
    reset_channels,
    retained_hashes,
    tensor_hash,
    zero_channels,
)
from .model import (
    GemmaAdapter,
    _set_identity,
    configure_runtime,
    load_native,
    parameter_counts,
    save_native,
)
from .training import resume_identity, train_records


class FixtureTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0
    special_tokens_map = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}

    def __init__(self):
        chars = ["\n"] + [chr(i) for i in range(32, 127)]
        self.chars = {char: i + 3 for i, char in enumerate(chars)}
        self.vocab = {
            **{f"<unused_{i}>": i for i in range(128)},
            **self.chars,
            "<pad>": 0,
            "<eos>": 1,
            "<bos>": 2,
        }
        self.inverse = {value: key for key, value in self.chars.items()}

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("Fixture expects explicit special tokens")
        try:
            return [self.chars[char] for char in text]
        except KeyError as exc:
            raise ValueError("Fixture tokenizer accepts ASCII plus newline only") from exc

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        result = []
        for token in ids:
            if token in (0, 1, 2):
                if not skip_special_tokens:
                    result.append({0: "<pad>", 1: "<eos>", 2: "<bos>"}[token])
            else:
                result.append(self.inverse.get(token, f"<unused_{token}>"))
        return "".join(result)

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def tiny_adapter(seed=42):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    configure_runtime("cpu", seed, fixture=True)
    config = Gemma3TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=64,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
        attention_dropout=0.0,
        initializer_range=0.02,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    model = Gemma3ForCausalLM(config).float().requires_grad_(False).eval()
    _set_identity(model)
    return GemmaAdapter(
        model,
        FixtureTokenizer(),
        {
            "base": None,
            "synthetic_seed": seed,
            "fixture_config_sha256": value_hash(config.to_dict()),
        },
        device="cpu",
        fixture=True,
    )


def tiny_recipe():
    return replace(
        Recipe(),
        max_length=128,
        native_dtype="float32",
        lora_rank=2,
        lora_alpha=4,
        fused_adamw=False,
        microbatch=2,
        gradient_accumulation=2,
        effective_batch_size=4,
        checkpoint_updates=1,
    )


def fixture_masks():
    masks = {"random": random_mask(3, 32, 8)}
    for selector, counts, offset in (
        ("taylor", [8] * 3, 0),
        ("guided", [8] * 3, 16),
        ("quota_free_guided", [0, 7, 17], 8),
    ):
        masks[selector] = {
            "format": MASK_FORMAT,
            "selector": selector,
            "original_width": 32,
            "layers": {
                str(i): list(range(offset, offset + count)) for i, count in enumerate(counts)
            },
        }
    return masks


def tiny_check(output, *, seed=42):
    from dataclasses import replace

    import torch

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Training seed must be a nonnegative 32-bit integer")
    output = new_output(output)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rows = [
            {
                "id": f"synthetic:{i}",
                "input": f"p{i}",
                "target": "A" * 17 if i == 0 else "A" * (i % 3 + 1),
            }
            for i in range(5)
        ]
        data_dir = output / "synthetic_data"
        data_dir.mkdir()
        payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
        zipped = data_dir / "records.jsonl.gz"
        with (
            zipped.open("xb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream,
        ):
            stream.write(payload)
        write_json(
            data_dir / "manifest.json",
            {
                "format": "remove-reset-data-v1",
                "fixture": True,
                "files": {
                    "records.jsonl.gz": {
                        "sha256": sha256(zipped),
                        "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
                        "records": len(rows),
                    }
                },
            },
        )
        resolved = DataRoot(data_dir).rows("records.jsonl.gz")
        if resolved != rows:
            raise ValueError("Portable fixture data resolution changed exact records")
        tokenizer = FixtureTokenizer()
        records = [encode_record(tokenizer, row, 128).validate() for row in resolved]
        recipe = replace(tiny_recipe(), seed=seed)
        masks = fixture_masks()
        summaries = []
        for arm in ARMS:
            directory = output / arm
            directory.mkdir()
            adapter = tiny_adapter()
            mask = (
                None
                if arm == "ordinary_lora"
                else masks[
                    "quota_free_guided"
                    if arm.startswith("quota_free_guided_")
                    else arm.split("_", 1)[0]
                ]
            )
            original = adapter.prompt_logits("x")
            if mask is not None:
                with zero_channels(adapter.network, mask, enabled=False):
                    identity_logits = adapter.prompt_logits("x")
                if not torch.equal(original, identity_logits):
                    raise ValueError("Tiny visited no-op failed")
                with zero_channels(adapter.network, mask):
                    zero = adapter.prompt_logits("x")
                before = retained_hashes(adapter.network, mask)
                if arm.endswith("_removal"):
                    remove_channels(adapter.network, mask)
                    after = retained_hashes(adapter.network)
                else:
                    reset_channels(adapter.network, mask)
                    after = retained_hashes(adapter.network, mask)
                if before != after:
                    raise ValueError("Tiny retained native bytes changed")
                delta = (zero - adapter.prompt_logits("x")).abs().max().item()
                if (arm.endswith("_reset") and delta != 0) or delta > 1e-5:
                    raise ValueError("Tiny zero-vs-intervention numerical control failed")
            native = directory / "native"
            native.mkdir()
            expected = adapter.prompt_logits("x")
            save_native(adapter, native, arm=arm, mask=mask)
            seal(native)
            adapter = load_native(native, None, device="cpu", fixture_tokenizer=tokenizer)
            if not torch.equal(expected, adapter.prompt_logits("x")):
                raise ValueError("Tiny safe native reload changed logits")
            bindings = {
                "synthetic": True,
                "records_sha256": value_hash(rows),
                "examples": len(records),
                "target_tokens": sum(r.target_tokens for r in records),
            }
            identity = resume_identity(adapter, recipe, bindings, arm=arm)
            frozen = [(p, tensor_hash(p)) for p in adapter.network.parameters()]
            run = directory / "training"
            run.mkdir()
            paused = train_records(
                adapter,
                records,
                recipe,
                run,
                identity,
                pause_at_update=1,
                reset_mask=mask if arm.endswith("_reset") else None,
            )
            if paused["updates"] != 1 or paused["complete"]:
                raise ValueError("Tiny pause did not stop at update one")
            if (
                paused["status"] != "paused"
                or paused["startup_controls"]["zero_update_lora_max_abs"] != 0
            ):
                raise ValueError("Tiny pause/zero-update controls reported incorrectly")
            if arm.endswith("_reset") and not all(
                (count > 0 and signal > 0) or (count == 0 and signal == 0)
                for count, signal in zip(
                    map(len, mask["layers"].values()),
                    paused["startup_controls"][
                        "selected_path_down_adapter_gradient_norms"
                    ],
                    strict=True,
                )
            ):
                raise ValueError("Tiny reset-path LoRA gradient signals differ by selected layer")
            for parameter, before_hash in frozen:
                if tensor_hash(parameter) != before_hash or parameter.grad is not None:
                    raise ValueError("Tiny LoRA training changed a frozen native weight")
            resumed = load_native(native, None, device="cpu", fixture_tokenizer=tokenizer)
            finished = train_records(resumed, records, recipe, run, identity, resume=True)
            if not finished["complete"] or finished["updates"] != 2 or finished["examples"] != 5:
                raise ValueError("Tiny resumed epoch did not consume the fixed full order")
            if arm == "quota_free_guided_removal":
                reference = load_native(native, None, device="cpu", fixture_tokenizer=tokenizer)
                reference_run = directory / "uninterrupted_reference"
                reference_run.mkdir()
                train_records(reference, records, recipe, reference_run, identity)
                a = {
                    n: p.detach() for n, p in resumed.network.named_parameters() if p.requires_grad
                }
                b = {
                    n: p.detach()
                    for n, p in reference.network.named_parameters()
                    if p.requires_grad
                }
                if a.keys() != b.keys() or any(not torch.equal(a[key], b[key]) for key in a):
                    raise ValueError("Tiny resumed/uninterrupted FP32 LoRA trajectory differs")
            counts = parameter_counts(resumed.network)
            resumed.freeze()
            first = resumed.generate("x", max_new_tokens=4, eos_token_ids=[1])
            repeat = resumed.generate("x", max_new_tokens=4, eos_token_ids=[1])
            if first["continuation_ids"] != repeat["continuation_ids"]:
                raise ValueError("Tiny repeated greedy continuation differs")
            write_json(directory / "generation.json", {"first": first, "repeat": repeat})
            summaries.append(
                {
                    "arm": arm,
                    "widths": resumed.widths,
                    **counts,
                    "updates": finished["updates"],
                    "tail_target_tokens": records[
                        training_order(len(records), seed, study=False)[-1]
                    ].target_tokens,
                    "native_reload_exact": True,
                    "repeat_greedy_exact": True,
                }
            )
        return save_result(
            output,
            ToolResult(
                title="Synthetic CPU end-to-end fixture",
                summary="All nine arm paths prepared, safe-reloaded, LoRA-trained, paused/resumed and generated.",
                metrics={
                    "fixture": True,
                    "training_seed": seed,
                    "arms": len(summaries),
                    "examples_per_arm": 5,
                    "updates_per_arm": 2,
                    "quota_free_guided_removal_resumed_vs_uninterrupted_exact": True,
                },
                tables={"arms": summaries},
                notes=[
                    "Randomly initialized 3-layer, width-32, vocab-128 FP32 Gemma on CPU.",
                    "Fixture LoRA r2/alpha4, effective batch4; NOT the frozen 4B CUDA recipe.",
                    "The one-example tail uses its actual target count; seed 42 has 19.",
                    "This checks code paths and fixture determinism, not historical 4B bitwise outputs, "
                    "training resource feasibility, benchmark accuracy or selection quality.",
                ],
            ),
        )
    except BaseException as exc:
        if not (output / "result.json").exists():
            save_result(
                output,
                ToolResult(title="Synthetic fixture failed", summary="No 4B claim."),
                status="failed",
                error=exc,
            )
        raise
    finally:
        torch.set_num_threads(old_threads)
