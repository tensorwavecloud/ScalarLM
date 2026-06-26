"""Resolve PEFT's "all-linear" target-modules shorthand ourselves.

PEFT's `LoraConfig(target_modules="all-linear")` is supposed to expand the
shorthand to every linear layer except the output head. We resolve it ourselves
for two reasons:

1. **MoE.** Under peft 0.19 + transformers 5.x the expansion silently fails for
   some architectures (observed for Qwen3MoeForCausalLM) and PEFT falls back to
   iterating the literal string as a *set of characters*, raising
   `Target modules {'-','l','n','r','i','a','e'} not found` — the `qwen3-moe`
   TRAIN_FAILED in the cuda-spark sweep.

2. **Multimodal.** For a `...ForConditionalGeneration` wrapper, "all-linear"
   would also adapt the vision encoder. PEFT matches `target_modules` by name
   *suffix across the whole model*, and a vision tower can reuse the language
   tower's leaf names (Gemma3's `vision_tower...self_attn.k_proj`), so a plain
   leaf-name set can't exclude it. We confine LoRA to the language decoder by
   emitting its Linear modules' *full paths*.

For dense models the result is the sorted leaf-name set — byte-identical to
PEFT's own expansion (same trainable parameters).
"""

import torch.nn as nn

# PEFT's sentinel for "adapt every linear layer except the output head".
ALL_LINEAR = "all-linear"


def _is_multimodal_model(model) -> bool:
    """True for HF multimodal wrappers — they nest a `vision_config` on their
    config. A `vision_config` present but `None` reads as not-multimodal."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    return getattr(config, "vision_config", None) is not None


def _language_decoder(model):
    """The text-decoder submodule to confine LoRA to on a multimodal model, or
    None when no scoping is needed (dense model, or no usable `get_decoder`).
    `get_decoder()` is the standard HF handle for the text tower
    (Gemma3TextModel, Qwen2VLTextModel)."""
    if not _is_multimodal_model(model):
        return None
    if not hasattr(model, "get_decoder"):
        return None
    decoder = model.get_decoder()
    if decoder is None or decoder is model:
        return None
    return decoder


def _module_prefix(model, target) -> str | None:
    """The dotted name of `target` within `model` (by identity), or None if it
    isn't found among the model's submodules."""
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


# Self-attention container segments across the common decoder architectures.
_ATTENTION_CONTAINERS = (".self_attn.", ".attn.", ".attention.")


def _is_moe_model(model) -> bool:
    """True if the model has routed MoE expert submodules (a `.experts` module).

    A `.pt` adapter that adapts the fused experts can't be served: vLLM's
    `FusedMoEWithLoRA.set_lora` wants a per-expert tensor *list* (gate/down/up,
    each `[num_experts, rank, dim]`), while the ScalarLM trainer exports stacked
    2-D tensors. Rather than reproduce vLLM's PEFT→fused-MoE conversion, we keep
    LoRA off the experts and adapt attention only — which serves cleanly."""
    return any(".experts" in name for name, _ in model.named_modules())


def _attention_linear_names(model, output_embeddings) -> set[str]:
    """Leaf names of the attention-projection `nn.Linear` modules (q/k/v/o under
    a self-attention container). Excludes MLP, MoE experts, the router, and the
    output head — everything that doesn't serve from a `.pt` MoE adapter."""
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        if any(seg in module_name for seg in _ATTENTION_CONTAINERS):
            names.add(module_name.split(".")[-1])
    return names


def resolve_target_modules(model, target_modules):
    """If `target_modules` is the "all-linear" shorthand, resolve it against the
    live `model`:

    - **multimodal** (config has `vision_config`, `get_decoder()` available):
      the full dotted paths of every `nn.Linear` under the language decoder,
      excluding the output head. PEFT matches these exactly, so a vision tower
      reusing the same leaf names is not adapted.
    - **MoE** (has routed `.experts` submodules, non-multimodal): the sorted
      attention-projection leaf names only. The fused-expert LoRA can't be served
      from a `.pt` adapter (vLLM's `FusedMoEWithLoRA` wants a per-expert tensor
      list, not stacked tensors), so LoRA is kept off the experts.
    - **dense**: the sorted set of distinct `nn.Linear` leaf-module names,
      excluding the output head — identical to PEFT's all-linear expansion.

    Any other value — an explicit list, a non-shorthand string, or None — is
    returned unchanged for PEFT to interpret itself.
    """
    if target_modules != ALL_LINEAR:
        return target_modules

    # Exclude the output projection. Prefer identity (handles heads not named
    # "lm_head"); fall back to the conventional leaf name below.
    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()

    decoder = _language_decoder(model)
    if decoder is not None:
        prefix = _module_prefix(model, decoder)
        if prefix is not None:
            return sorted(
                name
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
                and module is not output_embeddings
                and (name == prefix or name.startswith(prefix + "."))
            )
        # get_decoder() returned a module we couldn't locate — fall through to
        # the dense path rather than silently adapt nothing.

    if _is_moe_model(model):
        # Attention-only for MoE: the fused-expert LoRA isn't serveable from a
        # .pt adapter (see _is_moe_model), so confine LoRA to the attention
        # projections, which load + serve cleanly.
        attn = _attention_linear_names(model, output_embeddings)
        if attn:
            return sorted(attn)
        # No attention modules matched (unusual arch) — fall through to the dense
        # path rather than adapt nothing.

    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        names.add(module_name.split(".")[-1])
    names.discard("lm_head")  # belt-and-suspenders when get_output_embeddings()==None

    return sorted(names)
