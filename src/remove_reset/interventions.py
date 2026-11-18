"""Original-coordinate MLP masks, physical removal, and function-matched reset.

A channel is ONE gated-product coordinate: gate_proj/up_proj ROW j and
down_proj COLUMN j in the original width-10240 universe. Removal keeps the
ascending complement; reset retains full shapes and freezes ALL native weights.
No head, attention, neuron-output-only, or arbitrary module edit is inferred.
"""

from __future__ import annotations

from contextlib import contextmanager

from .artifacts import digest, read_json, sha256
from .config import QUOTA_FREE_COUNTS, arm_mask
from .contracts import Param, ToolResult, ToolSpec

MASK_FORMAT = "remove-reset-mask-v1"


def random_mask(layers, width, count, seed=1042):
    """Exact source algorithm: NumPy default_rng/PCG64, layer-ascending choice."""
    import numpy as np

    if any(type(x) is not int for x in (layers, width, count, seed)):
        raise ValueError("Mask dimensions and seed must be integers")
    if layers < 1 or not 0 < count < width or seed < 0:
        raise ValueError("A mask must select a nonempty proper subset in every layer")
    rng = np.random.default_rng(seed)
    return {
        "format": MASK_FORMAT,
        "selector": "random",
        "seed": seed,
        "original_width": width,
        "layers": {
            str(i): sorted(rng.choice(width, count, replace=False).tolist()) for i in range(layers)
        },
    }


def normalize_mask(mask, *, source_sha256=None):
    """Explicit transport conversion, NOT relaxation of the old source contracts.

    Shipped legacy JSON may be byte-identical to its original mask. Only these
    three documented formats are accepted. Their old selection digests remain
    provenance, never paths to an old run, approval, archive, or saved result.
    The release data manifest binds the shipped file's complete bytes.
    """
    if not isinstance(mask, dict):
        raise ValueError("A mask must be an object")
    form, selector = mask.get("format"), mask.get("selector")
    if form == MASK_FORMAT:
        if selector not in ("random", "taylor", "guided", "quota_free_guided"):
            raise ValueError(f"Unknown mask selector: {selector!r}")
    elif form == "remove-or-reset-pruned-v1":
        if selector not in ("random", "taylor", "guided"):
            raise ValueError("Unsupported legacy uniform selector")
        if type(mask.get("removed_per_layer")) is not int:
            raise ValueError("Legacy uniform mask needs its original per-layer count")
        if selector != "random":
            digest(mask.get("selection_sha256"), "original selection digest")
    elif form == "remove-or-reset-global-content-mask-v1":
        if selector != "global_mean_content":
            raise ValueError("Unsupported quota-free guided mask")
        selector = "quota_free_guided"
        digest(mask.get("selection_sha256"), "original quota-free selection digest")
        counts = mask.get("counts_by_layer")
        if (
            not isinstance(counts, list)
            or any(type(n) is not int for n in counts)
            or type(mask.get("total_selected")) is not int
            or mask["total_selected"] != sum(counts)
        ):
            raise ValueError("Legacy late allocation total differs")
    else:
        raise ValueError(f"Unsupported mask format: {form!r}")
    value = {
        "format": MASK_FORMAT,
        "selector": selector,
        "original_width": mask.get("original_width"),
        "layers": mask.get("layers"),
    }
    if selector == "random":
        value["seed"] = mask.get("seed", 1042)
    if source_sha256 is not None or "source_sha256" in mask:
        value["source_sha256"] = digest(source_sha256 or mask["source_sha256"])
    if form != MASK_FORMAT:
        for key, ids in (mask.get("layers") or {}).items():
            count = (
                mask["counts_by_layer"][int(key)]
                if selector == "quota_free_guided"
                else mask["removed_per_layer"]
            )
            if not isinstance(ids, list) or len(ids) != count:
                raise ValueError("Legacy mask counts and coordinates disagree")
    return value


def validate_mask(mask, layers=None, width=None, *, study=False):
    mask = normalize_mask(mask)
    width = mask["original_width"] if width is None else width
    if type(width) is not int or width < 2 or mask["original_width"] != width:
        raise ValueError("Invalid original channel universe")
    mapping = mask["layers"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Missing mask layers")
    layers = len(mapping) if layers is None else layers
    if type(layers) is not int or layers < 1 or set(mapping) != {str(i) for i in range(layers)}:
        raise ValueError("Mask must declare every original layer, exactly once")
    counts = []
    # JSON sort_keys produces 0,1,10,...,2,... . Allocation is NUMERICAL layer
    # order, never dictionary insertion/lexicographic order (important for late).
    for layer in range(layers):
        selected = mapping[str(layer)]
        if (
            not isinstance(selected, list)
            or not 0 <= len(selected) < width
            or any(type(j) is not int or not 0 <= j < width for j in selected)
            or selected != sorted(set(selected))
            or (not selected and mask["selector"] != "quota_free_guided")
        ):
            raise ValueError("Selected IDs must be sorted, unique, and in range")
        counts.append(len(selected))
    if mask["selector"] != "quota_free_guided" and len(set(counts)) != 1:
        raise ValueError("Uniform selectors require identical per-layer counts")
    if mask["selector"] == "random":
        expected = random_mask(layers, width, counts[0], mask["seed"])
        if mask["layers"] != expected["layers"]:
            raise ValueError("Random mask does not reproduce its seeded NumPy algorithm")
    if study:
        expected_counts = (
            list(QUOTA_FREE_COUNTS)
            if mask["selector"] == "quota_free_guided"
            else [2048] * 34
        )
        if (layers, width) != (34, 10240) or counts != expected_counts or sum(counts) != 69632:
            raise ValueError("Mask differs from the frozen study allocation")
        if mask["selector"] == "random" and mask["seed"] != 1042:
            raise ValueError("Study random mask seed must be 1042")
    return mask


def load_mask(data, selector):
    if selector not in ("random", "taylor", "guided", "quota_free_guided"):
        raise ValueError("Unsupported frozen mask")
    path = data.path(f"masks/{selector}.json")
    mask = normalize_mask(read_json(path), source_sha256=sha256(path))
    mask = validate_mask(mask, study=True)
    if mask["selector"] != selector:
        raise ValueError("Mask filename and selector disagree")
    return mask


def mask_for_arm(data, arm):
    name = arm_mask(arm)
    return None if name is None else load_mask(data, name)


def complement(mask):
    return {
        key: [j for j in range(mask["original_width"]) if j not in set(ids)]
        for key, ids in mask["layers"].items()
    }


def validate_native_shapes(model, mask):
    layers, width = len(model.model.layers), model.config.intermediate_size
    mask = validate_mask(mask, layers, width)
    for block in model.model.layers:
        for name, shape in (
            ("gate_proj", (width, model.config.hidden_size)),
            ("up_proj", (width, model.config.hidden_size)),
            ("down_proj", (model.config.hidden_size, width)),
        ):
            old = getattr(block.mlp, name)
            if old.bias is not None or tuple(old.weight.shape) != shape:
                raise ValueError("Intervention requires original bias-free gated MLP shapes")
    return mask


def install_shapes(model, widths):
    """Meta-safe constructor, original config universe PLUS actual layer widths.

    Never set a single misleading intermediate_size for a ragged model. All
    expected state keys/shapes are constructed before any safe tensor is loaded.
    This function is not Hugging Face's ignore_mismatched_sizes path.
    """
    import torch

    if (
        not isinstance(widths, list)
        or len(widths) != len(model.model.layers)
        or any(type(w) is not int or not 0 < w <= model.config.intermediate_size for w in widths)
    ):
        raise ValueError("Invalid explicit compact widths")
    for block, width in zip(model.model.layers, widths, strict=True):
        for name in ("gate_proj", "up_proj", "down_proj"):
            old = getattr(block.mlp, name)
            if old.bias is not None:
                raise ValueError("Only bias-free Gemma MLPs are supported")
            ins = width if name == "down_proj" else old.in_features
            outs = old.out_features if name == "down_proj" else width
            setattr(
                block.mlp,
                name,
                torch.nn.Linear(
                    ins, outs, bias=False, device=old.weight.device, dtype=old.weight.dtype
                ),
            )
        block.mlp.intermediate_size = width
    model.config.compact_widths = list(widths)
    model.config.remove_reset_format = "remove-reset-native-v1"
    return model


def remove_channels(model, mask):
    """Physically slice gate/up rows and down columns, retaining ascending IDs."""
    import torch

    mask = validate_native_shapes(model, mask)
    kept = complement(mask)
    widths = []
    for layer, block in enumerate(model.model.layers):
        indices = torch.tensor(kept[str(layer)], device=block.mlp.down_proj.weight.device)
        widths.append(len(indices))
        for name, dimension in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
            weight = (
                getattr(block.mlp, name)
                .weight.detach()
                .index_select(dimension, indices)
                .contiguous()
            )
            new = torch.nn.Linear(
                weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=weight.dtype
            )
            new.weight = torch.nn.Parameter(weight, requires_grad=False)
            setattr(block.mlp, name, new)
        block.mlp.intermediate_size = len(indices)
    # Keep config.intermediate_size == ORIGINAL coordinate universe.
    model.config.compact_widths = widths
    model.config.remove_reset_format = "remove-reset-native-v1"
    model.requires_grad_(False)
    return model


def reset_channels(model, mask, seed=3042):
    """CPU seed 3042, layer-ascending gate then up FP32 N(0,.02) -> native dtype.

    Selected down columns are exactly zero. Draw order and tensor shapes matter:
    late allocation uses the same generator, NOT matching draws on shared IDs.
    Unselected entries are checked bitwise; no native parameter becomes trainable.
    """
    import torch

    if type(seed) is not int or seed < 0:
        raise ValueError("Reset seed must be a nonnegative integer")
    mask = validate_native_shapes(model, mask)
    if model.config.initializer_range != 0.02:
        raise ValueError("Unexpected linear initializer standard deviation")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kept = complement(mask)
    edits = []
    with torch.no_grad():
        for layer, block in enumerate(model.model.layers):
            for name, dimension in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                weight = getattr(block.mlp, name).weight
                indices = torch.tensor(
                    mask["layers"][str(layer)], dtype=torch.long, device=weight.device
                )
                retained = torch.tensor(
                    kept[str(layer)], dtype=torch.long, device=weight.device
                )
                before_kept = weight.index_select(dimension, retained).clone()
                if dimension == 0:
                    samples = torch.empty(
                        (len(indices), weight.shape[1]), dtype=torch.float32, device="cpu"
                    )
                    samples.normal_(mean=0, std=0.02, generator=generator)
                    weight.index_copy_(
                        0, indices, samples.to(device=weight.device, dtype=weight.dtype)
                    )
                else:
                    weight.index_fill_(1, indices, 0)
                if not torch.equal(before_kept, weight.index_select(dimension, retained)):
                    raise ValueError("Reset changed an unselected native weight")
                selected = weight.index_select(dimension, indices)
                if not torch.isfinite(selected).all() or (
                    dimension == 1 and torch.count_nonzero(selected).item()
                ):
                    raise ValueError("Reset produced nonfinite values/nonzero outgoing columns")
                edits.append(
                    {
                        "layer": layer,
                        "projection": name,
                        "selected_entries": selected.numel(),
                        "unselected_exact": True,
                    }
                )
    model.requires_grad_(False)
    return edits


@contextmanager
def zero_channels(model, mask, *, enabled=True):
    """Actual gated-product input to down_proj, all positions; explicit scale-1 control."""
    import torch

    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    mask = validate_native_shapes(model, mask)
    visits = [0] * len(model.model.layers)
    handles = []
    failed = False
    try:
        for layer, block in enumerate(model.model.layers):
            indices = torch.tensor(
                mask["layers"][str(layer)],
                dtype=torch.long,
                device=block.mlp.down_proj.weight.device,
            )

            def hook(module, inputs, layer=layer, indices=indices):
                if len(inputs) != 1 or inputs[0].ndim != 3:
                    raise ValueError("Unexpected down-projection input shape")
                visits[layer] += 1
                if not enabled:
                    return inputs
                hidden = inputs[0].clone()
                hidden.index_fill_(-1, indices, 0)
                return (hidden,)

            handles.append(block.mlp.down_proj.register_forward_pre_hook(hook))
        yield visits
    except BaseException:
        failed = True
        raise
    finally:
        for handle in handles:
            handle.remove()
        if not failed and any(v == 0 for v in visits):
            raise ValueError("A requested intervention site was never visited")


def tensor_hash(tensor):
    import hashlib

    import torch

    result = hashlib.sha256()
    for chunk in tensor.detach().reshape(-1).split(4 * 1024 * 1024):
        result.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def retained_hashes(model, mask=None):
    """Hash all retained parameters; before removal, project to the same complement."""
    import torch

    kept = None if mask is None else complement(mask)
    result = {}
    for name, value in model.state_dict().items():
        if kept is not None and ".mlp." in name and name.endswith(".weight"):
            layer = name.split(".")[2]
            dimension = 1 if ".down_proj." in name else 0
            value = value.index_select(dimension, torch.tensor(kept[layer], device=value.device))
        result[name] = tensor_hash(value)
    return result


def _mask_audit(ctx, params):
    import numpy as np

    path = params["mask"]
    mask = validate_mask(
        normalize_mask(read_json(path), source_sha256=sha256(path)), study=params["study"]
    )
    width = mask["original_width"]
    rows = [
        {
            "layer": layer,
            "selected": len(mask["layers"][str(layer)]),
            "retained": width - len(mask["layers"][str(layer)]),
            "original_selected_ids": mask["layers"][str(layer)],
        }
        for layer in range(len(mask["layers"]))
    ]
    selected = np.zeros((len(rows), width), dtype=bool)
    for row in rows:
        selected[row["layer"], row["original_selected_ids"]] = True
    return ToolResult(
        title="Frozen mask coordinate audit",
        summary="Counts and validates original MLP coordinates without loading any weights.",
        metrics={
            "selector": mask["selector"],
            "layers": len(rows),
            "original_width": width,
            "selected_channels": int(selected.sum()),
            "mask_file_sha256": sha256(path),
        },
        tables={"allocation": rows},
        notes=[
            "A coordinate couples gate/up rows and the matching down column.",
            "Counts are intervention metadata, not evidence that selected channels are unimportant.",
            "This audit checks masks, not the effects of removal/reset or historical model outputs.",
        ],
        arrays={"selected_original_coordinates": selected},
    )


def get_tools():
    return (
        ToolSpec(
            id="mask-audit",
            title="Frozen mask coordinate audit",
            description="Strict original-coordinate/allocation audit; no model or GPU required.",
            parameters=(
                Param("mask", "text", "", help="Local mask JSON file"),
                Param("study", "boolean", True, help="Enforce the 34-layer frozen allocation"),
            ),
            run=_mask_audit,
        ),
    )


def get_tool(name):
    for tool in get_tools():
        if tool.id == name:
            return tool
    raise ValueError(f"Unknown tool: {name}")
