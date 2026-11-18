"""One offline text-only Gemma adapter; no arbitrary model/pickle/remote-code loading.

All analysis/training/evaluation uses this adapter. Native parameters are BF16
on CUDA in the study, LoRA parameters are FP32. The deliberately separate tiny
fixture is FP32/CPU and is NEVER an automatic production fallback.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

from .artifacts import (
    read_json,
    sha256,
    source_identity,
    text_hash,
    value_hash,
    verify_bundle,
    write_json,
)
from .config import MODEL_ID, PROJECTIONS, REVISION, arm_mask
from .contracts import ModelInfo, Trace


def finite_tensor(tensor, name):
    """Bound temporary validation memory even for the vocabulary embedding."""
    import torch

    if tensor.device.type == "meta":
        raise ValueError(f"Unloaded meta tensor: {name}")
    for chunk in tensor.detach().reshape(-1).split(4 * 1024 * 1024):
        if not torch.isfinite(chunk).all().item():
            raise ValueError(f"Nonfinite tensor: {name}")


def snapshot_directory(base_dir):
    """Accept an explicit local snapshot or HF cache root; never call a downloader."""
    root = Path(base_dir).resolve(strict=True)
    if (root / "config.json").is_file():
        return root
    candidate = root / "models--google--gemma-3-4b-pt" / "snapshots" / REVISION
    if not (candidate / "config.json").is_file():
        raise ValueError("Pinned snapshot missing; obtain it manually after accepting its license")
    return candidate


def base_binding(base_dir):
    directory = snapshot_directory(base_dir)
    files = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(directory.iterdir())
        if p.is_file() and (p.suffix in (".json", ".model", ".safetensors"))
    }
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Local pinned safetensors weights are required")
    return {"model_id": MODEL_ID, "revision": REVISION, "files": files}


def configure_runtime(device="cuda:0", seed=42, *, fixture=False):
    import os
    import random

    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if not fixture:
        if (
            not str(device).startswith("cuda")
            or not torch.cuda.is_available()
            or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Study requires CUDA BF16; no CPU/offload/quantization fallback")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def parameter_counts(model):
    counts = {"native_parameters": 0, "adapter_parameters": 0, "trainable_parameters": 0}
    for name, parameter in model.named_parameters():
        kind = "adapter_parameters" if "lora_" in name else "native_parameters"
        counts[kind] += parameter.numel()
        if parameter.requires_grad:
            counts["trainable_parameters"] += parameter.numel()
    return counts


def _checkpoint_key(name):
    for prefix in ("model.language_model.", "language_model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _constructor(config, widths, dtype):
    """Match native construction dtype while keeping FP32 rotary frequencies.

    init_empty_weights(include_buffers=False) leaves Gemma's nonpersistent
    rotary buffers materialized by their actual FP32 formula. A blanket
    model.to(dtype=BF16) would change them and is deliberately never used.
    """
    import torch
    from accelerate import init_empty_weights
    from transformers import Gemma3ForCausalLM

    from .interventions import install_shapes

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with init_empty_weights(include_buffers=False):
            model = Gemma3ForCausalLM(config)
            if widths is not None:
                model = install_shapes(model, widths)
    finally:
        torch.set_default_dtype(previous)
    return model


def strict_load(directory, config, widths, *, device, dtype, official=False):
    """Header-first complete key/shape/dtype/index validation, then safe assignment.

    Source vision tensors are explicitly excluded ONLY for the official
    multimodal snapshot. A native checkpoint permits no unexplained tensors.
    Per-shard strict=False assignment is safe here because the COMPLETE expected
    state was validated first; it is not permission to initialize missing weights.
    """
    import torch
    from safetensors import safe_open

    directory = Path(directory)
    model = _constructor(config, widths, dtype)
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    mapped, all_shards, skipped = {}, {}, []
    allowed_vision = (
        "vision_tower.",
        "multi_modal_projector.",
        "model.vision_tower.",
        "model.multi_modal_projector.",
    )
    required_dtype = "BF16" if dtype == torch.bfloat16 else "F32"
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise ValueError("No safetensors weights; pickle fallback is prohibited")
    for path in shards:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in all_shards:
                    raise ValueError(f"Duplicate source tensor: {name}")
                all_shards[name] = path.name
                key = _checkpoint_key(name) if official else name
                if key not in expected:
                    if official and name.startswith(allowed_vision):
                        skipped.append(name)
                        continue
                    raise ValueError(f"Unexpected native tensor: {name}")
                if key in mapped:
                    raise ValueError(f"Duplicate adapted text key: {key}")
                view = stream.get_slice(name)
                if tuple(view.get_shape()) != expected[key] or view.get_dtype() != required_dtype:
                    raise ValueError(f"Native shape/dtype mismatch: {key}")
                mapped[key] = (name, path.name)
    missing = set(expected) - set(mapped)
    allowed = (
        {"lm_head.weight"}
        if config.tie_word_embeddings and ("model.embed_tokens.weight" in mapped)
        else set()
    )
    if missing - allowed:
        raise ValueError(f"Missing native text keys: {sorted(missing - allowed)}")
    index = directory / "model.safetensors.index.json"
    if index.exists():
        if read_json(index).get("weight_map") != all_shards:
            raise ValueError("Safetensors shard index differs from actual headers")
    elif len(shards) != 1:
        raise ValueError("Multiple safetensors shards require an exact index")
    for path in shards:
        with safe_open(str(path), framework="pt", device=str(torch.device(device))) as stream:
            values = {}
            for key, (original, filename) in mapped.items():
                if filename == path.name:
                    tensor = stream.get_tensor(original)
                    finite_tensor(tensor, key)
                    values[key] = tensor
            model.load_state_dict(values, strict=False, assign=True)
    if config.tie_word_embeddings and "lm_head.weight" in mapped:
        if not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
            raise ValueError("Stored tied embedding aliases disagree")
    model.tie_weights()
    if any(p.device.type == "meta" for p in model.parameters()):
        raise ValueError("Unassigned native model parameter")
    # Move buffers only in device, never blanket-cast their precision.
    model.to(device=device).requires_grad_(False).eval()
    if config.tie_word_embeddings and model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("Tied embedding identity was not preserved")
    for name, parameter in model.named_parameters():
        if parameter.dtype != dtype or parameter.device != torch.device(device):
            raise ValueError(f"Unexpected native dtype/device: {name}")
    for name, buffer in model.named_buffers():
        if "inv_freq" in name and buffer.dtype != torch.float32:
            raise ValueError("Rotary frequency precision changed")
    return model, {
        "mapped_keys": len(mapped),
        "excluded_vision_keys": skipped,
        "missing_tied_aliases": sorted(missing),
    }


class GemmaAdapter:
    """Dedicated ModelAdapter plus tensor-efficient training/generation methods.

    encode() adds NO special tokens; data.encode_record/generate() insert exactly
    one BOS explicitly. trace() is a bounded observational method (<=64 tokens)
    and refuses every activation edit rather than accepting ignored interventions.
    Channel interventions use the separate explicit original-coordinate contract.
    """

    def __init__(self, network, tokenizer, provenance, *, device="cuda:0", fixture=False):
        self.network = network
        self.tokenizer = tokenizer
        self.provenance = provenance
        self.device = str(device)
        self.fixture = fixture
        config = network.config
        self.info = ModelInfo(
            name=MODEL_ID if not fixture else "synthetic-tiny-gemma-fixture",
            n_layers=config.num_hidden_layers,
            d_model=config.hidden_size,
            d_mlp=config.intermediate_size,
            vocab_size=config.vocab_size,
            max_seq_len=2048 if not fixture else config.max_position_embeddings,
            n_parameters=sum(p.numel() for p in network.parameters()),
            device=self.device,
            fingerprint=value_hash(provenance),
            revision=REVISION if not fixture else "synthetic",
            extra={"fixture": fixture, "widths": self.widths},
        )

    @property
    def base(self):
        return (
            self.network.get_base_model()
            if hasattr(self.network, "get_base_model")
            else self.network
        )

    @property
    def widths(self):
        return [block.mlp.down_proj.in_features for block in self.base.model.layers]

    def encode(self, text):
        if not isinstance(text, str) or "\x00" in text:
            raise ValueError("Expected text without NUL")
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        self._validate_ids(ids)
        return ids

    def _validate_ids(self, ids):
        if any(type(x) is not int or not 0 <= x < self.info.vocab_size for x in ids):
            raise ValueError("Token ID outside actual embedding vocabulary")

    def decode(self, ids: Sequence[int]):
        self._validate_ids(ids)
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def target_logits(self, ids, attention, positions):
        import torch

        if (
            ids.ndim != 2
            or attention.shape != ids.shape
            or positions.ndim != 1
            or positions.dtype != torch.long
            or not positions.numel()
            or int(positions.min()) < 0
            or int(positions.max()) >= ids.shape[1]
            or ids.shape[1] > self.info.max_seq_len
        ):
            raise ValueError("Invalid input/attention/causal-position shape or context length")
        scope = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with scope:
            logits = self.network(
                input_ids=ids, attention_mask=attention, use_cache=False, logits_to_keep=positions
            ).logits
        if logits.shape != (ids.shape[0], len(positions), self.info.vocab_size):
            raise ValueError("Model did not honor exact tensor logits_to_keep")
        return logits

    def trace(self, prompt, interventions=()):
        import torch

        if interventions:
            raise ValueError("Gemma trace does not support activation interventions")
        ids = self.encode(prompt)
        if not 1 <= len(ids) <= min(64, self.info.max_seq_len):
            raise ValueError("Observational trace requires 1..64 tokens, without implicit BOS")
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.target_logits(
                tensor, torch.ones_like(tensor), torch.arange(len(ids), device=self.device)
            )
        return Trace([self.decode([i]) for i in ids], ids, logits[0].detach().cpu(), {})

    def project(self, residual):
        import torch

        if (
            residual.ndim not in (1, 2)
            or residual.shape[-1] != self.info.d_model
            or not torch.isfinite(residual).all()
        ):
            raise ValueError("Projection expects finite [width] or [token,width] residual")
        dtype = self.base.model.norm.weight.dtype
        with torch.no_grad():
            hidden = self.base.model.norm(residual.to(device=self.device, dtype=dtype))
            logits = self.base.get_output_embeddings()(hidden)
            cap = self.base.config.final_logit_softcapping
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap
        return logits.detach().cpu()

    def prompt_logits(self, text):
        import torch

        ids = [self.tokenizer.bos_token_id] + self.encode(text)
        if len(ids) > self.info.max_seq_len:
            raise ValueError("Untruncated prompt exceeds context")
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return (
                self.target_logits(
                    tensor,
                    torch.ones_like(tensor),
                    torch.tensor([len(ids) - 1], device=self.device),
                )[0, 0]
                .float()
                .cpu()
            )

    def generate(
        self,
        prompt,
        *,
        max_new_tokens,
        eos_token_ids,
        skip_special_tokens=False,
        expected_prompt_ids=None,
        seed=42,
    ):
        """Batch-one plain-BOS greedy generation; explicit highest-priority settings.

        Disables model-default inheritance, verifies effective greedy_search and
        EOS, and preserves the exact raw token continuation (including stop token).
        SQL/reasoning use EOS [1]; heldout uses [1,106]. No chat template,
        sampling, prefix text matching, stop-string edits, or answer salvage.
        """
        import time

        import torch
        from transformers import GenerationConfig

        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
            or not isinstance(eos_token_ids, list)
            or not eos_token_ids
            or any(type(x) is not int for x in eos_token_ids)
        ):
            raise ValueError("Invalid generation budget/EOS IDs")
        self._validate_ids(eos_token_ids)
        ids = [self.tokenizer.bos_token_id] + self.encode(prompt)
        if expected_prompt_ids is not None and ids != expected_prompt_ids:
            raise ValueError("Actual prompt IDs differ from the frozen plan")
        if len(ids) + max_new_tokens > self.info.max_seq_len:
            raise ValueError("Generation context would exceed limit; no truncation")
        if self.network.training or any(
            p.requires_grad or p.grad is not None for p in self.network.parameters()
        ):
            raise ValueError("Generation requires eval and all parameters frozen/gradient-free")
        torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        explicit = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "cache_implementation": "dynamic",
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_token_ids,
        }
        configuration = GenerationConfig(**explicit)
        effective, _ = self.network._prepare_generation_config(
            configuration, use_model_defaults=False, **explicit
        )
        if (
            effective.do_sample is not False
            or effective.num_beams != 1
            or effective.max_new_tokens != max_new_tokens
            or effective.eos_token_id != eos_token_ids
            or effective.get_generation_mode().value != "greedy_search"
        ):
            raise ValueError("Effective generation configuration is not the fixed greedy protocol")
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.network.generate(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                generation_config=configuration,
                use_model_defaults=False,
                logits_to_keep=1,
                **explicit,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError("Unexpected generation output shape")
        all_ids = output[0].detach().cpu().tolist()
        if all_ids[: len(ids)] != ids:
            raise ValueError("Generation changed the prompt prefix")
        continuation = all_ids[len(ids) :]
        if (
            not 1 <= len(continuation) <= max_new_tokens
            or any(x in eos_token_ids for x in continuation[:-1])
            or (continuation[-1] not in eos_token_ids and len(continuation) != max_new_tokens)
        ):
            raise ValueError("Invalid continuation length/stop contract")
        raw = self.tokenizer.decode(
            continuation,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        return {
            "prompt_ids": ids,
            "prompt_ids_sha256": value_hash(ids),
            "prompt_tokens": len(ids),
            "continuation_ids": continuation,
            "continuation_ids_sha256": value_hash(continuation),
            "continuation_tokens": len(continuation),
            "raw_answer": raw,
            "raw_answer_utf8_sha256": text_hash(raw),
            "stop_status": "eos" if continuation[-1] in eos_token_ids else "max_new_tokens",
            "effective_do_sample": False,
            "effective_generation_mode": "greedy_search",
            "use_model_defaults": False,
            "eos_token_ids": eos_token_ids,
            "effective_generation_config_sha256": value_hash(effective.to_dict()),
            "elapsed_seconds": elapsed,
        }

    def freeze(self):
        self.network.requires_grad_(False).eval()
        for parameter in self.network.parameters():
            parameter.grad = None


def load_base(base_dir, *, device="cuda:0"):
    import torch
    from transformers import AutoTokenizer, Gemma3TextConfig

    if str(device).startswith("cuda") and torch.device(device).index is None:
        device = f"cuda:{torch.cuda.current_device()}"
    configure_runtime(device)
    directory = snapshot_directory(base_dir)
    original = read_json(directory / "config.json")
    if (
        original.get("model_type") != "gemma3"
        or "text_config" not in original
        or original.get("quantization_config")
    ):
        raise ValueError("Expected the pinned, unquantized official Gemma3 multimodal config")
    config = Gemma3TextConfig.from_dict(original["text_config"])
    config._attn_implementation = "sdpa"
    if (config.num_hidden_layers, config.hidden_size, config.intermediate_size) != (
        34,
        2560,
        10240,
    ):
        raise ValueError("Pinned text architecture differs")
    tokenizer = AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False
    )
    model, diagnostics = strict_load(
        directory, config, None, device=device, dtype=torch.bfloat16, official=True
    )
    _validate_tokenizer(model, tokenizer)
    _set_identity(model)
    if parameter_counts(model)["native_parameters"] != 3880263168:
        raise ValueError("Pinned native parameter count differs")
    provenance = {
        "base": base_binding(directory),
        "text_only": True,
        "loading": diagnostics,
        "parameter_counts": parameter_counts(model),
    }
    return GemmaAdapter(model, tokenizer, provenance, device=device)


def _set_identity(model):
    model.name_or_path = MODEL_ID
    model.config._name_or_path = MODEL_ID
    model.config._commit_hash = REVISION


def _validate_tokenizer(model, tokenizer):
    vocabulary = tokenizer.get_vocab()
    if (
        not vocabulary
        or min(vocabulary.values()) < 0
        or max(vocabulary.values()) >= model.config.vocab_size
        or tokenizer.bos_token_id != model.config.bos_token_id
        or tokenizer.eos_token_id != model.config.eos_token_id
        or tokenizer.pad_token_id is None
    ):
        raise ValueError("Tokenizer/embedding/config vocabulary or BOS/EOS/pad mismatch")


def attach_lora(adapter, recipe):
    import torch
    from peft import LoraConfig, get_peft_model

    model = adapter.network
    if hasattr(model, "peft_config"):
        raise ValueError("Cannot attach a second adapter")
    expected = {
        name
        for name, module in model.named_modules()
        if name.split(".")[-1] in PROJECTIONS and isinstance(module, torch.nn.Linear)
    }
    if len(expected) != len(model.model.layers) * 7:
        raise ValueError("Every layer must expose exactly seven declared projection types")
    model.requires_grad_(False)
    model.config.use_cache = False
    # Explicit reset after deterministic native construction matches the source
    # PEFT initialization seed. Reset uses its own CPU Generator, not this RNG.
    torch.manual_seed(recipe.seed)
    if adapter.device.startswith("cuda"):
        torch.cuda.manual_seed_all(recipe.seed)
    model = get_peft_model(
        model,
        LoraConfig(
            base_model_name_or_path=MODEL_ID,
            revision=REVISION,
            task_type="CAUSAL_LM",
            r=recipe.lora_rank,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            target_modules=list(recipe.target_modules),
            bias="none",
            init_lora_weights=True,
        ),
    )
    actual = {name for name, module in model.named_modules() if hasattr(module, "lora_A")}
    if len(actual) != len(expected) or not all(
        any(name.endswith(old) for name in actual) for old in expected
    ):
        raise ValueError("PEFT did not cover every declared native projection")
    native_dtype = torch.float32 if adapter.fixture else torch.bfloat16
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"Unexpected trainable native parameter: {name}")
            parameter.data = parameter.data.float()
        elif parameter.dtype != native_dtype:
            raise ValueError(f"Wrong frozen native dtype: {name}")
        if parameter.device != torch.device(adapter.device):
            raise ValueError("LoRA/native offload is not supported")
    if recipe.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    adapter.network = model
    if not adapter.fixture:
        counts = parameter_counts(model)
        expected_native = 3345489408 if min(adapter.widths) < 10240 else 3880263168
        expected_adapter = 26460160 if expected_native == 3345489408 else 29802496
        if counts != {
            "native_parameters": expected_native,
            "adapter_parameters": expected_adapter,
            "trainable_parameters": expected_adapter,
        }:
            raise ValueError("Study native/FP32 LoRA parameter count differs")
    return adapter


def save_native(adapter, directory, *, arm, mask, data_manifest_sha256=None):
    """Reader-generated native weights only; tokenizer always stays in the base cache."""
    from .interventions import normalize_mask

    directory = Path(directory)
    if hasattr(adapter.network, "peft_config"):
        raise ValueError("Native export must precede attaching LoRA")
    adapter.network.save_pretrained(str(directory), safe_serialization=True, max_shard_size="2GB")
    if mask is not None:
        write_json(directory / "mask.json", normalize_mask(mask))
    write_json(
        directory / "native.json",
        {
            "format": "remove-reset-native-v1",
            "arm": arm,
            "fixture": adapter.fixture,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "base": adapter.provenance.get("base"),
            "original_width": adapter.network.config.intermediate_size,
            "compact_widths": adapter.widths,
            "parameter_counts": parameter_counts(adapter.network),
            "mask_sha256": sha256(directory / "mask.json") if mask is not None else None,
            "data_manifest_sha256": data_manifest_sha256,
            "source_identity": source_identity(),
        },
    )


def load_native(directory, base_dir, *, device="cuda:0", fixture_tokenizer=None):
    import torch
    from transformers import AutoTokenizer, Gemma3TextConfig

    from .interventions import validate_mask

    verify_bundle(directory)
    directory = Path(directory)
    native = read_json(directory / "native.json")
    fixture = fixture_tokenizer is not None
    if (
        native.get("format") != "remove-reset-native-v1"
        or native.get("fixture") is not fixture
        or (native["model_id"], native["revision"]) != (MODEL_ID, REVISION)
    ):
        raise ValueError("Native checkpoint identity/fixture contract differs")
    expected_selector = arm_mask(native["arm"])
    if native["source_identity"] != source_identity():
        raise ValueError("Native preparation source differs; reprepare with the current source")
    if str(device).startswith("cuda") and torch.device(device).index is None:
        device = f"cuda:{torch.cuda.current_device()}"
    configure_runtime(device, fixture=fixture)
    if fixture:
        tokenizer = fixture_tokenizer
    else:
        if native["base"] != base_binding(base_dir):
            raise ValueError("Supplied base/tokenizer files differ from native preparation")
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_directory(base_dir)), local_files_only=True, trust_remote_code=False
        )
    config = Gemma3TextConfig.from_dict(read_json(directory / "config.json"))
    config._attn_implementation = "sdpa"
    if not fixture:
        original = Gemma3TextConfig.from_dict(
            read_json(snapshot_directory(base_dir) / "config.json")["text_config"]
        )
        semantics = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "sliding_window",
            "layer_types",
            "rope_scaling",
            "rope_theta",
            "rms_norm_eps",
            "attention_bias",
            "attention_dropout",
            "hidden_activation",
            "final_logit_softcapping",
            "query_pre_attn_scalar",
            "attn_logit_softcapping",
            "initializer_range",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        )
        if any(getattr(config, key, None) != getattr(original, key, None) for key in semantics):
            raise ValueError(
                "Native computation/config differs from the pinned original text model"
            )
    widths = native["compact_widths"]
    if config.intermediate_size != native["original_width"]:
        raise ValueError("Native original universe differs from config")
    if native["mask_sha256"] is not None:
        if sha256(directory / "mask.json") != native["mask_sha256"]:
            raise ValueError("Native mask binding changed")
        mask = validate_mask(read_json(directory / "mask.json"), study=not fixture)
        if mask["selector"] != expected_selector:
            raise ValueError("Native arm and mask selector disagree")
        expected = [
            mask["original_width"] - len(mask["layers"][str(i)])
            if native["arm"].endswith("_removal")
            else mask["original_width"]
            for i in range(len(mask["layers"]))
        ]
        if widths != expected:
            raise ValueError("Physical widths do not follow the bound mask/arm")
    elif native["arm"] not in ("ordinary_lora", "base"):
        raise ValueError("Intervened native checkpoint must bind a mask")
    dtype = torch.float32 if fixture else torch.bfloat16
    model, diagnostics = strict_load(directory, config, widths, device=device, dtype=dtype)
    if parameter_counts(model) != native["parameter_counts"]:
        raise ValueError("Reloaded native parameter accounting differs")
    if not fixture:
        expected_native = 3345489408 if native["arm"].endswith("_removal") else 3880263168
        if parameter_counts(model)["native_parameters"] != expected_native:
            raise ValueError("Study native parameter count differs")
    _validate_tokenizer(model, tokenizer)
    _set_identity(model)
    return GemmaAdapter(
        model,
        tokenizer,
        {
            "base": native["base"],
            "native": native,
            "native_manifest_sha256": sha256(directory / "manifest.json"),
            "loading": diagnostics,
        },
        device=device,
        fixture=fixture,
    )
