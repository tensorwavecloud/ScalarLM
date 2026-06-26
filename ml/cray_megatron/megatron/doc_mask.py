"""Decide how to handle packed-document attention masking per batch/model.

The trainer packs short documents into one block and builds a 4D
block-diagonal+causal attention mask ``[B, 1, S, S]`` so packed documents don't
attend across each other (see ``pack()`` in
``dataset/load_language_model_dataset.py`` and ``training_step_accumulate`` in
``training_loop.py``). Plain text decoders (Llama, Qwen, Gemma3ForCausalLM)
accept the 4D mask.

But some ``...ForConditionalGeneration`` wrappers compute their loss internally
and index the logits by a *2D* attention_mask. Gemma3ForConditionalGeneration
does ``shift_logits[shift_attention_mask != 0]`` (transformers
``modeling_gemma3.py``); handed a 4D mask that index is 4D and raises
``IndexError: too many indices for tensor of dimension 3`` — the gemma-3-4b-it
TRAIN_FAILED in the cuda-spark fine-tune sweep. For those models we fall back to
the 2D padding mask. The documented cost is that packed documents attend across
each other (identical to the existing seq-len-cap fallback); for the
single-document memorization sweep the 4D mask is a no-op anyway.
"""

# Decision outcomes for doc_mask_decision().
BUILD = "build"                    # construct the 4D block-diagonal+causal mask
SKIP_MULTIMODAL = "skip_multimodal"  # wrapper masks loss by a 2D mask; keep 2D
SKIP_SEQLEN = "skip_seqlen"        # mask too large to materialize; keep 2D
NONE = "none"                      # batch isn't packed (no document_ids)


def is_multimodal(model_config) -> bool:
    """True for HF multimodal wrapper configs (Gemma3/Qwen2-VL/Gemma4, …) — they
    nest a ``vision_config``. A ``vision_config`` attribute that is present but
    ``None`` reads as not-multimodal."""
    if model_config is None:
        return False
    return getattr(model_config, "vision_config", None) is not None


def doc_mask_decision(batch, seq_len: int, model_config, max_4d_mask_seq_len: int) -> str:
    """Return how to handle packed-document attention for this batch:

    - ``NONE``            — the batch isn't packed (no ``document_ids``).
    - ``SKIP_MULTIMODAL`` — a multimodal wrapper that masks its loss by a 2D
      attention_mask; a 4D mask would break that index. Keep the 2D mask.
    - ``SKIP_SEQLEN``     — the mask would exceed ``max_4d_mask_seq_len``; keep
      the 2D mask (legacy fallback).
    - ``BUILD``           — construct the 4D block-diagonal+causal mask.

    The multimodal check takes precedence over the seq-len cap: both fall back
    to the 2D mask, but the multimodal reason is the one worth surfacing.
    """
    if "document_ids" not in batch:
        return NONE
    if is_multimodal(model_config):
        return SKIP_MULTIMODAL
    if seq_len > max_4d_mask_seq_len:
        return SKIP_SEQLEN
    return BUILD
