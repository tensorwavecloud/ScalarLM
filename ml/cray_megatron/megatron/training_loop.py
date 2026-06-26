from cray_megatron.megatron.dataset.data_loader import DataLoader

from cray_megatron.models.get_model_manager import get_model_manager
from cray_megatron.models.does_any_checkpoint_exist import does_any_checkpoint_exist
from cray_megatron.models.get_latest_checkpoint_path import (
    get_latest_checkpoint_path,
    delete_old_checkpoints,
)

from cray_megatron.collectives.main_rank_only import main_rank_only, is_main_rank
from cray_megatron.megatron.training_harness import TrainingHarness
from cray_megatron.megatron import stop_flag
from cray_megatron.megatron.doc_mask import (
    doc_mask_decision,
    is_multimodal,
    BUILD,
    SKIP_SEQLEN,
    SKIP_MULTIMODAL,
)

from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.util.get_job_config import get_job_config


from torch.optim import AdamW

import torch

import time
import logging
from gpu_aware_mpi import allreduce, get_size

try:
    from torch.nn.attention.flex_attention import BlockMask
    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    _FLEX_ATTENTION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Cap on sequence length for the SDPA path's materialized [B, 1, S, S]
# block-diagonal document mask. Bool mask is 1 byte/cell, so 16384 →
# ~256 MiB at batch=1, comfortable; 128K → 16 GB, fatal.
# Use attn_implementation="flex_attention" in train_args to lift this
# cap: the flex path uses a BlockMask that is O(S/128), not O(S²).
_MAX_4D_MASK_SEQ_LEN = 16384
_doc_mask_skip_warned = False
_doc_mask_multimodal_warned = False


def _warn_doc_mask_skipped(seq_len: int) -> None:
    global _doc_mask_skip_warned
    if _doc_mask_skip_warned:
        return
    _doc_mask_skip_warned = True
    logger.warning(
        "Skipping 4D document mask: seq_len=%d exceeds cap %d. "
        "Packed documents will attend across each other in this run. "
        "Set attn_implementation='flex_attention' in train_args to lift this limit.",
        seq_len, _MAX_4D_MASK_SEQ_LEN,
    )


def _warn_doc_mask_skipped_multimodal() -> None:
    global _doc_mask_multimodal_warned
    if _doc_mask_multimodal_warned:
        return
    _doc_mask_multimodal_warned = True
    logger.warning(
        "Skipping 4D document mask: multimodal wrapper masks its loss by a 2D "
        "attention_mask (e.g. Gemma3ForConditionalGeneration indexes shift_logits "
        "by it), so a 4D mask raises IndexError. Falling back to the 2D mask; "
        "packed documents will attend across each other in this run."
    )


def _use_flex_attention() -> bool:
    return get_job_config().get("attn_implementation") == "flex_attention"


def _build_document_block_mask(doc_ids, device):
    """Build BlockMask for packed-document causal attention without O(S²) materialization.

    doc_ids: [B, S] monotonically-increasing per-doc integer tag (from the packer).

    create_block_mask (even with _compile=True) calls create_mask which vmaps
    mask_mod over all S² positions, materializing the full [B,H,S,S] bool tensor
    (16 GB at 128K). This function builds the same BlockMask directly from
    block-level doc-ID range overlap in O((S/BLOCK)²) memory — ~1 MB at 128K.
    """
    try:
        from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as BLOCK
    except ImportError:
        BLOCK = 128

    B, S = doc_ids.shape
    num_blocks = (S + BLOCK - 1) // BLOCK
    padded_S   = num_blocks * BLOCK

    # Pad to block boundary with a sentinel doc_id that never matches any real doc.
    if padded_S > S:
        sentinel   = int(doc_ids.max().item()) + 1
        pad        = torch.full((B, padded_S - S), sentinel, dtype=doc_ids.dtype, device=device)
        doc_ids_p  = torch.cat([doc_ids, pad], dim=1)
    else:
        doc_ids_p = doc_ids

    # Per-block doc-ID range: min and max of the BLOCK tokens in each block.
    d     = doc_ids_p.view(B, num_blocks, BLOCK)
    d_min = d.min(dim=-1).values   # [B, num_blocks]
    d_max = d.max(dim=-1).values   # [B, num_blocks]

    blk = torch.arange(num_blocks, device=device)

    # Causal at block level: kv_b <= q_b means there exists at least one valid
    # (q_pos, kv_pos) pair satisfying q_pos >= kv_pos.
    causal_any      = blk.view(1, -1) <= blk.view(-1, 1)   # [num_q, num_kv]
    strictly_causal = blk.view(1, -1) <  blk.view(-1, 1)   # [num_q, num_kv]

    # Doc overlap: ranges [d_min_kv, d_max_kv] and [d_min_q, d_max_q] intersect.
    d_min_q  = d_min.unsqueeze(2)   # [B, num_q, 1]
    d_max_q  = d_max.unsqueeze(2)   # [B, num_q, 1]
    d_min_kv = d_min.unsqueeze(1)   # [B, 1, num_kv]
    d_max_kv = d_max.unsqueeze(1)   # [B, 1, num_kv]
    doc_overlap = (d_min_kv <= d_max_q) & (d_min_q <= d_max_kv)   # [B, num_q, num_kv]

    # non_empty: block (q_b, kv_b) has at least one valid attention connection.
    non_empty = causal_any.unsqueeze(0) & doc_overlap   # [B, num_q, num_kv]

    # is_full: EVERY position in kv_block can attend to EVERY position in q_block
    # without needing per-token masking.  Requires kv strictly before q (no causal
    # boundary within the pair) and both blocks entirely within the same single doc.
    q_single  = (d_min_q  == d_max_q)    # [B, num_q, 1]
    kv_single = (d_min_kv == d_max_kv)   # [B, 1, num_kv]
    same_doc  = (d_min_q  == d_min_kv)   # [B, num_q, num_kv]
    is_full   = strictly_causal.unsqueeze(0) & q_single & kv_single & same_doc

    def _pack(mask_bqkv):
        """Pack a [B, nq, nkv] bool mask into (counts [B,1,nq], indices [B,1,nq,max_k])."""
        Bm, nq, nkv = mask_bqkv.shape
        counts = mask_bqkv.sum(dim=2).to(torch.int32)   # [B, nq]
        max_k  = int(counts.max().item()) if counts.numel() > 0 else 0
        if max_k == 0:
            return counts.unsqueeze(1), torch.zeros(Bm, 1, nq, 0, dtype=torch.int32, device=device)
        all_idx = torch.arange(nkv, device=device).view(1, 1, nkv).expand(Bm, nq, nkv)
        # Replace invalid entries with sentinel nkv so they sort after valid ones.
        filled  = torch.where(mask_bqkv, all_idx, torch.full_like(all_idx, nkv))
        sorted_idx, _ = filled.sort(dim=2)
        # Take the first max_k entries; clamp out-of-range padding to a safe index.
        idx = sorted_idx[:, :, :max_k].clamp(0, nkv - 1).to(torch.int32)
        return counts.unsqueeze(1), idx.unsqueeze(1)

    kv_num_blocks,      kv_indices      = _pack(non_empty)
    full_kv_num_blocks, full_kv_indices = _pack(is_full)
    q_num_blocks,       q_indices       = _pack(non_empty.transpose(1, 2))
    full_q_num_blocks,  full_q_indices  = _pack(is_full.transpose(1, 2))

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    try:
        return BlockMask(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            BLOCK_SIZE=(BLOCK, BLOCK),
            mask_mod=mask_mod,
            seq_lengths=(S, S),
        )
    except TypeError:
        # seq_lengths not present in older PyTorch builds.
        return BlockMask(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            BLOCK_SIZE=(BLOCK, BLOCK),
            mask_mod=mask_mod,
        )


class TrainingLoop:
    def __init__(self, training_harness: TrainingHarness):
        self.training_harness = training_harness

        self.callbacks = get_callbacks(self)

        self.training_state = TrainingState()

    def train(self):
        self.model_manager = get_model_manager()

        self.training_state.model_info = self.model_manager.load_model()

        self.training_loop()

        self.checkpoint()

        self._finalize_slice()

    def _finalize_slice(self):
        """Persist accumulated wall time after every slice. Carried
        forward via status.json so the next slice's TimeoutCallback
        compares against the user's TOTAL budget, not just this slice.
        See docs/training-lifecycle.md §5.4.

        Slice-end dispatch (queuing the next slice when slurm cuts a
        slice short) is handled separately by `restart_megatron_jobs`
        on the API server: it polls every megatron_refresh_period for
        TRAINING/QUEUED jobs missing from squeue and re-submits via
        start_slurm_job. MegatronTrainer flips to COMPLETED only when
        no signal was received, which is what keeps that reconciler
        from respawning jobs that finished cleanly."""
        slice_elapsed = time.time() - self.training_state.start_time
        accumulated = (
            self.training_state.accumulated_seconds_at_slice_start + slice_elapsed
        )
        self._persist_accumulated_seconds(accumulated)

    @main_rank_only
    def _persist_accumulated_seconds(self, accumulated_seconds):
        # Read-modify-write through the harness preserves status,
        # job_id, history, etc. We never change status here — that's
        # MegatronTrainer's job (or update_history's during the loop).
        current = self.training_harness.get_status()
        self.training_harness.update_status(
            status=current.get("status", TrainingJobStatus.TRAINING),
            metadata={"accumulated_train_seconds": accumulated_seconds},
        )

    def training_loop(self):
        self.on_train_begin()

        self._load_accumulated_seconds()

        self.training_state.model_info["model"].train()

        max_steps = get_max_steps()
        gradient_accumulation_steps = get_gradient_accumulation_steps()

        self.training_state.optimizer = get_optimizer(
            self.training_state.model_info["model"]
        )
        self.training_state.scheduler = get_scheduler(
            self.training_state.optimizer, max_steps
        )

        if does_any_checkpoint_exist():
            self.resume_from_checkpoint()

        data_loader = DataLoader(
            model=self.training_state.model_info["model"],
            tokenizer=self.training_state.model_info["tokenizer"],
            starting_epoch=self.training_state.epoch,
        )

        data_iterator = iter(data_loader)

        # Fast-forward to the saved batch position. IterableDataset has
        # no seek, so we pull and discard `data_cursor` batches. This is
        # slow (full tokenize + pack pipeline runs on each one) but still
        # much cheaper than re-training the steps from scratch — and
        # combined with the restored RNG state below, reproduces exactly
        # the batch stream the previous slice was about to consume.
        if self.training_state.data_cursor > 0:
            logger.info(
                "Resuming dataloader: skipping %d batches into epoch %d",
                self.training_state.data_cursor,
                self.training_state.epoch,
            )
            for _ in range(self.training_state.data_cursor):
                next(data_iterator)

        starting_step = self.training_state.current_step

        self.print_device_info()

        for step in range(starting_step, max_steps):
            self.training_state.current_step = step
            self.training_state.epoch = data_loader.epoch

            step_start_time = time.time()

            self.on_step_begin(step)

            # Accumulate gradients over multiple micro-batches
            accumulated_loss = 0.0
            has_nan = False
            self.training_state.optimizer.zero_grad()

            for accum_step in range(gradient_accumulation_steps):
                prev_epoch = data_loader.epoch
                batch = next(data_iterator)

                # Update epoch if it changed during accumulation
                self.training_state.epoch = data_loader.epoch

                # Track cursor for checkpoint resume. Reset to 1 (this
                # batch) on epoch rollover, otherwise increment. The
                # rollover happens inside next() when the iterator
                # exhausts and DataLoader.__next__ swaps in a new dataset
                # for epoch+1, so the comparison sees prev_epoch < new.
                if data_loader.epoch != prev_epoch:
                    self.training_state.data_cursor = 1
                else:
                    self.training_state.data_cursor += 1

                loss = self.training_step_accumulate(
                    batch,
                    accum_step,
                    gradient_accumulation_steps
                )

                # Check for NaN or Inf. torch.isnan misses ±inf, which can
                # propagate through backward, survive clip_grad_norm (per-tensor
                # norm of an inf tensor is inf and clip clamps it to max_norm
                # in finite arithmetic), and corrupt the next optimizer step.
                if not torch.isfinite(torch.tensor(loss)):
                    has_nan = True
                    logger.warning(
                        f"Non-finite loss detected at step {step}, microbatch {accum_step + 1} (loss={loss})"
                    )
                    break

                accumulated_loss += loss

            # Skip optimizer step if NaN was detected
            if has_nan:
                logger.warning(
                    f"Skipping optimizer step {step} due to NaN loss"
                )
                self.training_state.nan_steps += 1
                avg_accumulated_loss = float('nan')
            else:
                # Average the accumulated loss
                avg_accumulated_loss = accumulated_loss / gradient_accumulation_steps

                # Ensure gradients are synchronized across ranks during backward pass
                self.training_state.model_info["model"].backward_sync()

                # Perform optimizer step after accumulation
                self.optimizer_step()

                # Log the averaged loss
                self.update_history(avg_accumulated_loss)

            # Print training step info with averaged loss
            step_time = time.time() - step_start_time
            self.print_training_step_info(avg_accumulated_loss, step_time)

            self.on_step_end(step)

            if stop_flag.was_stop_requested():
                logger.info(
                    "Stop requested via signal %s — exiting training loop "
                    "at step %d to checkpoint cleanly",
                    stop_flag.last_signal(),
                    step,
                )
                self.training_state.should_stop_training = True

            if self.training_state.should_stop_training:
                break

        self.on_train_end()

    def _load_accumulated_seconds(self):
        # Sum of wall-time across prior slices. Persisted in status.json
        # by _finalize_slice; TimeoutCallback adds the current slice's
        # in-flight elapsed to this when checking the user's total
        # budget.
        status = self.training_harness.get_status()
        self.training_state.accumulated_seconds_at_slice_start = float(
            status.get("accumulated_train_seconds", 0.0)
        )
        if self.training_state.accumulated_seconds_at_slice_start > 0:
            logger.info(
                "Resuming with %.0fs of prior training already elapsed",
                self.training_state.accumulated_seconds_at_slice_start,
            )

    def resume_from_checkpoint(self):
        latest_checkpoint_path = get_latest_checkpoint_path()
        logger.info(f"Resuming from checkpoint {latest_checkpoint_path}")

        checkpoint = torch.load(latest_checkpoint_path, weights_only=True)

        # `step` in the checkpoint is the step that *completed* when the
        # save fired (CheckpointCallback runs in on_step_end). Start the
        # loop at the next step so we don't redo the saved step — and,
        # because the saved step lands on a steps_per_checkpoint
        # boundary, redoing it would also trigger an immediate
        # re-checkpoint at on_step_end.
        self.training_state.current_step = checkpoint["step"] + 1
        self.training_state.epoch = checkpoint["epoch"]
        self.training_state.nan_steps = checkpoint.get("nan_steps", 0)
        # .get() with default=0 keeps older checkpoints (pre-Fix 2) loadable.
        self.training_state.data_cursor = checkpoint.get("data_cursor", 0)
        model = self.training_state.model_info["model"]
        if hasattr(model, "load_unwrapped_model"):
            # FSDP: the checkpoint holds all-gathered full tensors, but the
            # live model is sharded. Re-shard each tensor and write this rank's
            # slice in (the inverse of unwrap_model). Mirrors the save-side
            # branch in checkpoint() that dispatches on unwrap_model.
            model.load_unwrapped_model(checkpoint["model_state_dict"])
        else:
            _load_trained_parameters(model, checkpoint["model_state_dict"])
        self.training_state.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        self.training_state.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        # Restore RNG so the next forward (dropout, LoRA dropout, any
        # other stochastic op) is bit-identical to what would have come
        # next without the timeout. CPU tensor goes back to CPU via .cpu()
        # — torch.set_rng_state insists on a CPU ByteTensor.
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state.cpu())
        cuda_rng_state = checkpoint.get("cuda_rng_state")
        if cuda_rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_rng_state])

        self.training_state.history = self.training_harness.get_status()["history"]

    def training_step_accumulate(self, batch, accum_step, gradient_accumulation_steps):
        """Perform a single forward/backward pass with gradient accumulation."""
        device = self.training_state.model_info["distribution_strategy"]["device"]

        start_time = time.time()

        forward_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }

        # If the packed batch carries document_ids, replace the 1-D
        # attention_mask with a 4-D block-diagonal+causal additive mask
        # so packed documents don't attend across each other. The 1-D
        # mask only flags padding; passing it as-is makes the model
        # treat the whole packed block as one causal sequence, which
        # mixes unrelated documents into every attention head. See pack()
        # in load_language_model_dataset.py for the upstream change.
        # Capped at _MAX_4D_MASK_SEQ_LEN to keep memory bounded — above
        # that the mask is skipped and we fall back to legacy behavior
        # (a flash-attn varlen path is the right long-term fix).
        seq_len = forward_kwargs["input_ids"].shape[-1]
        model_config = self.training_state.model_info.get("model_config")
        decision = doc_mask_decision(batch, seq_len, model_config, _MAX_4D_MASK_SEQ_LEN)
        use_flex = _use_flex_attention() and _FLEX_ATTENTION_AVAILABLE
        if decision == SKIP_MULTIMODAL:
            # A ...ForConditionalGeneration wrapper whose internal loss indexes
            # the logits by a 2D attention_mask (gemma-3-4b-it). Keep the 2D mask
            # from the batch — a 4D mask (SDPA or flex BlockMask) would IndexError.
            _warn_doc_mask_skipped_multimodal()
        elif decision == BUILD or (decision == SKIP_SEQLEN and use_flex):
            # BUILD, or a sequence past the SDPA cap when flex_attention is on —
            # the flex BlockMask is O(S/128), so _MAX_4D_MASK_SEQ_LEN doesn't apply.
            doc_ids = batch["document_ids"].to(device)
            if use_flex:
                # flex_attention path: O(S/128) BlockMask, no sequence-length cap.
                # Requires attn_implementation="flex_attention" in train_args so the
                # model kernel and the mask format match.
                forward_kwargs["attention_mask"] = _build_document_block_mask(doc_ids, device)
            else:
                # SDPA path: materialize the [B, 1, S, S] causal block-diagonal
                # bool mask (1 byte/cell, capped at _MAX_4D_MASK_SEQ_LEN).
                same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
                causal = torch.ones(
                    seq_len, seq_len, device=device, dtype=torch.bool
                ).tril()
                forward_kwargs["attention_mask"] = (same_doc & causal).unsqueeze(1)
            forward_kwargs["position_ids"] = batch["position_ids"].to(device)
        elif decision == SKIP_SEQLEN:
            _warn_doc_mask_skipped(seq_len)
        # decision == NONE: batch isn't packed; leave the 2D mask as-is.

        # Multimodal wrappers (Gemma4ForConditionalGeneration, …) require an
        # mm_token_type_ids tensor on every training forward, even when the
        # batch is pure text. The data loader produces text-only batches; pass
        # zeros so non-image tokens are tagged correctly. Detected by the
        # presence of a vision_config on the underlying model config.
        if is_multimodal(model_config):
            forward_kwargs["mm_token_type_ids"] = torch.zeros_like(
                forward_kwargs["input_ids"]
            )

        # forward pass
        loss = self.training_state.model_info["model"](**forward_kwargs).loss

        # Scale loss to account for accumulation
        scaled_loss = loss / gradient_accumulation_steps

        # Synchronize loss across all ranks
        _, avg_loss = self.sync_loss(loss)

        # Always call backward, even when loss is NaN/Inf. backward()'s
        # job we care about here is walking the autograd graph and
        # freeing the saved-for-backward activations — gradient
        # computation is just a side effect. Skipping backward on NaN
        # leaks the entire forward graph (full activations under
        # gradient checkpointing's recompute pattern), and gc.collect()
        # +empty_cache() can't reclaim it because the graph is held by
        # strong refs (autograd Node ctx, FSDP allgather closures), not
        # cycles.
        #
        # The NaN gradients land on `param.grad`, but optimizer.step()
        # is skipped at step level via `has_nan` and the next step's
        # zero_grad() clears them before any weight update — weights
        # are protected without touching backward. (Substituting a
        # finite loss before backward — `loss * 0`, `where(isnan, 0,
        # loss)`, etc. — doesn't help: NaN propagates through the
        # already-saved activations, since the kernels' backward ops
        # multiply the seed gradient by NaN intermediates and IEEE
        # 754 makes `0 * NaN = NaN`.)
        scaled_loss.backward()

        # Log info for each micro-batch
        self.print_microbatch_info(accum_step, avg_loss, start_time)

        return avg_loss

    def optimizer_step(self):
        """Perform optimizer and scheduler step after gradient accumulation."""
        torch.nn.utils.clip_grad_norm_(
            self.training_state.model_info["model"].parameters(),
            get_gradient_clip_value(),
        )
        self.training_state.optimizer.step()
        self.training_state.scheduler.step()

    def sync_loss(self, loss):
        device = self.training_state.model_info["distribution_strategy"]["device"]
        if get_size() > 1:
            local_loss = allreduce_op(loss)
            avg_loss = local_loss.item() / get_size()
        else:
            avg_loss = loss.item()

        return loss, avg_loss

    def on_train_begin(self):
        self.training_state.start_time = time.time()
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin()

    def on_step_begin(self, step):
        for callback in self.callbacks:
            if hasattr(callback, "on_step_begin"):
                callback.on_step_begin(step)

    def on_step_end(self, step):
        for callback in self.callbacks:
            if hasattr(callback, "on_step_end"):
                callback.on_step_end(step)

    def on_train_end(self):

        logger.info(
            f"Training finished successfully after {time.time() - self.training_state.start_time} seconds"
        )
        if self.training_state.nan_steps > 0:
            logger.warning(
                f"Encountered {self.training_state.nan_steps} NaN steps during training"
            )
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end()

    def checkpoint(self):
        model = self.training_state.model_info["model"]
        model_state_dict = {}
        if hasattr(model, "unwrap_model"):
            logger.info("Unwrapping model")
            model_state_dict = self.training_state.model_info["model"].unwrap_model()
        else:
            model_state_dict = filter_checkpoint(model.model, model.model.state_dict())

        self.save_checkpoint(model_state_dict)

    @main_rank_only
    def save_checkpoint(self, model_state_dict):

        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.training_state.optimizer.state_dict(),
            "scheduler_state_dict": self.training_state.scheduler.state_dict(),
            "step": self.training_state.current_step,
            "epoch": self.training_state.epoch,
            "nan_steps": self.training_state.nan_steps,
            "data_cursor": self.training_state.data_cursor,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "metadata": build_adapter_metadata(),
        }

        checkpoint_name = f"checkpoint_{self.training_state.current_step}.pt"

        self.training_harness.checkpoint(
            checkpoint_state=checkpoint,
            checkpoint_name=checkpoint_name,
        )

        delete_old_checkpoints()

    @main_rank_only
    def update_history(self, loss):
        job_config = get_job_config()

        max_history_length = job_config["training_history_length"]

        entry = {
            "step": self.training_state.current_step,
            "loss": loss,
            # Total wall-clock across ALL slices, not just this one. start_time
            # resets every slice (on_train_begin), so the bare
            # `time.time() - start_time` is only the current slice's elapsed —
            # which is why the UI's elapsed and list_models' train_time both
            # reset to ~0 on every checkpoint resume. Add the wall time carried
            # forward from prior slices (loaded in _load_accumulated_seconds,
            # persisted in _finalize_slice) so the displayed elapsed survives
            # checkpoints. See docs/training-lifecycle.md §5.4.
            "epoch": self.training_state.epoch,
            "time": (
                self.training_state.accumulated_seconds_at_slice_start
                + (time.time() - self.training_state.start_time)
            ),
        }

        self.training_state.history.append(entry)

        if len(self.training_state.history) > max_history_length:
            self.training_state.history = remove_closest_entry(
                self.training_state.history, max_history_length
            )

        self.training_harness.update_status(
            status=TrainingJobStatus.TRAINING,
            metadata={"history": self.training_state.history},
        )

    @main_rank_only
    def print_training_step_info(self, loss, step_time):
        logger.info(
            f"Training step {self.training_state.current_step} "
            f"- epoch {self.training_state.epoch} "
            f"- avg loss {loss:.4f} "
            f"- learning rate {self.training_state.scheduler.get_last_lr()[0]:.6f} "
            f"- step time {step_time:.3f} seconds"
        )

    @main_rank_only
    def print_microbatch_info(self, accum_step, loss, start_time):
        # only log if there is more than one microbatch
        if get_gradient_accumulation_steps() <= 1:
            return

        logger.debug(
            f"  Microbatch {accum_step + 1} "
            f"- step {self.training_state.current_step} "
            f"- loss {loss:.4f} "
            f"- time {time.time() - start_time:.3f}s"
        )

    def print_device_info(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        idx = self.training_state.model_info["distribution_strategy"]["device"]
        logger.info(f"Using device {device}:{idx}")


def get_callbacks(trainer):
    return [
        TimeoutCallback(trainer),
        CheckpointCallback(trainer),
        CudaMemoryCallback(trainer),
    ]


class TimeoutCallback:
    """Stops training when the user's TOTAL budget (`train_args["timeout"]`)
    is exhausted across all slurm slices, not just the current slice.
    Single-slice trainers and pre-relaunch trainers behave identically
    because accumulated_seconds_at_slice_start defaults to 0."""

    def __init__(self, trainer):
        self.trainer = trainer
        job_config = get_job_config()
        self.timeout = job_config["timeout"]
        self.start_time = time.time()

    def on_train_begin(self):
        pass

    def on_step_end(self, step):
        slice_elapsed = time.time() - self.start_time
        total_elapsed = (
            self.trainer.training_state.accumulated_seconds_at_slice_start
            + slice_elapsed
        )
        if total_elapsed > self.timeout:
            logger.info(
                "Training timed out after %.0fs total (%.0fs prior slices + "
                "%.0fs current slice) — user budget was %.0fs",
                total_elapsed,
                self.trainer.training_state.accumulated_seconds_at_slice_start,
                slice_elapsed,
                self.timeout,
            )
            self.trainer.training_state.should_stop_training = True


class CheckpointCallback:
    def __init__(self, trainer):
        self.trainer = trainer
        job_config = get_job_config()
        self.steps_per_checkpoint = job_config["steps_per_checkpoint"]

    def on_step_end(self, step):
        if step % self.steps_per_checkpoint == 0 and step != 0:
            start_time = time.time()
            self.trainer.checkpoint()
            logger.info(
                f"Checkpoint on step {step} took {time.time() - start_time} seconds"
            )


class CudaMemoryCallback:
    """
    Periodic snapshot of CUDA allocator state. The two numbers split a
    leak diagnosis: `allocated` is live tensor bytes, `reserved` is
    what PyTorch is holding from the driver (active + cached free
    blocks). Reserved climbing while allocated stays flat is
    fragmentation, not a leak. Logged every `cuda_memory_log_interval`
    steps to keep the training log readable.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        job_config = get_job_config()
        self.interval = job_config.get("cuda_memory_log_interval", 100)

    @main_rank_only
    def on_step_end(self, step):
        if self.interval <= 0 or step % self.interval != 0:
            return
        if not torch.cuda.is_available():
            return
        gib = 1024 ** 3
        allocated = torch.cuda.memory_allocated() / gib
        reserved = torch.cuda.memory_reserved() / gib
        max_allocated = torch.cuda.max_memory_allocated() / gib
        logger.info(
            f"CUDA memory @ step {step}: "
            f"allocated={allocated:.2f} GiB, "
            f"reserved={reserved:.2f} GiB, "
            f"max_allocated={max_allocated:.2f} GiB"
        )


class TrainingState:
    def __init__(self):
        self.should_stop_training = False
        self.current_step = 0
        self.epoch = 0
        self.model_info = None
        self.optimizer = None
        self.scheduler = None
        self.history = []
        self.start_time = None
        self.nan_steps = 0
        # Batches consumed in the current epoch. Saved in the checkpoint
        # and restored on resume so the dataloader fast-forwards to the
        # exact batch the previous slice was on, instead of replaying the
        # first ~N batches of epoch 0 on every restart. Resets to 1 on
        # the batch that triggers an epoch rollover; counter sits at a
        # step boundary at checkpoint time because CheckpointCallback
        # fires on_step_end.
        self.data_cursor = 0
        # Loaded from status.json at the start of every slice (0 on the
        # first slice). TimeoutCallback adds the in-flight slice elapsed
        # to this when checking the user's total budget.
        self.accumulated_seconds_at_slice_start = 0.0


def get_max_steps():
    job_config = get_job_config()
    return job_config["max_steps"]


def get_gradient_accumulation_steps():
    job_config = get_job_config()
    return job_config.get("gradient_accumulation_steps", 4)


def get_optimizer(model):
    job_config = get_job_config()
    learning_rate = job_config["learning_rate"]
    # Only optimize trainable parameters. With LoRA adapters, PEFT marks all
    # frozen base-model weights as requires_grad=False, so this restricts AdamW
    # to the adapter parameters only (~100 MB fp32 states vs ~16 GB for a 2B
    # full-parameter AdamW).
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Initializing optimizer for {len(trainable)} trainable parameters")
    return AdamW(trainable, lr=learning_rate)

    # use Adafactor optimizer
    # return torch.optim.Adafactor(
    #    model.parameters(),
    #    lr=learning_rate,
    # )

def get_gradient_clip_value():
    job_config = get_job_config()
    return job_config.get("gradient_clip_value", 1.0)


def get_warmup_steps():
    job_config = get_job_config()
    return int(job_config.get("warmup_steps", 0))


def get_scheduler(optimizer, max_steps):
    warmup_steps = get_warmup_steps()
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max_steps,
        )

    # Cold optimizer state + full LR at step 0 is a known instability
    # source on large LoRA fine-tunes (see Fix 4 in the stability audit).
    # Ramp from lr/1000 to lr over warmup_steps, then linear decay to 0
    # over the remaining budget. start_factor=1e-3 (not 0) avoids the
    # "first optimizer.step uses LR=0" footgun where Adam's bias-corrected
    # moments still update from a zero-LR step.
    decay_steps = max(1, max_steps - warmup_steps)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    decay = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=decay_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
    )


def remove_closest_entry(history, max_length):
    # The training history should include evenly spaced entries
    # up to the maximum length
    # Remove the most closely spaced entry to another entry
    # until the history is the correct length
    while len(history) > max_length:
        min_diff = float("inf")
        min_index = None
        for i in range(1, len(history) - 1):
            diff = history[i + 1]["step"] - history[i - 1]["step"]
            if diff < min_diff:
                min_diff = diff
                min_index = i
        history.pop(min_index)

    return history


def filter_checkpoint(model, state_dict):
    # Remove the layers without gradients
    saved_params = {}

    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            logger.info(f"Saving parameter {name}")
            saved_params[name] = state_dict[name]

    return saved_params


def _load_trained_parameters(wrapped_model, state_dict):
    """Load a filtered (trainable-only) checkpoint back into the live model.

    The checkpoint is written relative to an *inner* module: the
    no-distribution / DDP path saves ``filter_checkpoint(model.model, ...)``
    (keys relative to the HF ``…ForCausalLM`` — depth 1) and the LoRA path
    saves relative to the PEFT base model (depth 2); FSDP saves via
    ``unwrap_model()``. The handle we hold on resume, though, is the *outer*
    distribution wrapper (NoDistribution / DDPLayer / FSDP / PEFT), whose own
    ``state_dict`` keys carry one or more extra ``model.``-style prefixes.

    Calling ``wrapper.load_state_dict(state_dict, strict=False)`` therefore
    matches *nothing* and — because ``strict=False`` silently tolerates the
    mismatch — drops every trained weight on the floor. That is the
    checkpoint-boundary loss-jump bug: optimizer / scheduler / RNG restore
    correctly, but the weights snap back to their ``from_pretrained`` values
    at each resume, so the loss spikes and has to re-descend.

    Resolve the descendant module the checkpoint keys actually belong to
    (walking ``.model`` a few levels), load there, and fail loudly if the
    namespaces still don't line up — never silently no-op again.
    """
    # Candidate load targets: the wrapper and its `.model` descendants.
    candidates = []
    module = wrapped_model
    seen = set()
    for _ in range(4):
        if module is None or id(module) in seen:
            break
        seen.add(id(module))
        candidates.append(module)
        module = getattr(module, "model", None)

    checkpoint_keys = set(state_dict)

    # Pick the target whose own parameter namespace overlaps the checkpoint
    # the most. The module the keys were saved from has all of them; any
    # other level (extra or missing `model.` prefix) overlaps with ~none.
    best_target = None
    best_matched = -1
    for candidate in candidates:
        matched = len(checkpoint_keys & set(candidate.state_dict().keys()))
        if matched > best_matched:
            best_matched = matched
            best_target = candidate

    if best_matched <= 0:
        raise RuntimeError(
            f"Checkpoint resume could not align any of the {len(checkpoint_keys)} "
            f"saved parameter(s) with the live model. The model_state_dict "
            f"namespace matches no module in "
            f"{type(wrapped_model).__name__}(.model)*. Sharded FSDP checkpoints "
            f"need the reshard-on-load path; for the single-process / DDP / LoRA "
            f"paths this is a wrapper-namespace regression."
        )

    incompatible = best_target.load_state_dict(state_dict, strict=False)

    # missing_keys is expected — we deliberately save only trainable params,
    # so every frozen base weight lands here. unexpected_keys must be empty:
    # a non-empty list means saved trained weights had no home in the target
    # and were dropped, which is the silent-corruption we're guarding against.
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint resume left {len(incompatible.unexpected_keys)} trained "
            f"parameter(s) unloaded (unexpected keys), e.g. "
            f"{incompatible.unexpected_keys[:3]}. Refusing to continue with "
            f"partially-restored weights."
        )

    logger.info(
        "Restored %d trained parameter tensor(s) into %s on resume",
        best_matched,
        type(best_target).__name__,
    )
    return best_target


def build_adapter_metadata():
    """
    Build the `metadata` dict that goes into the saved `.pt` alongside
    `model_state_dict`. Consumed by the inference-side adapter loader —
    see vllm/docs/training/adapter_format.md for the contract.

    For LoRA adapters we must emit `lora_alpha` explicitly. When it's
    missing the loader defaults to `2 * rank`, which silently mis-scales
    the delta by 2× whenever the trainer used `lora_alpha == rank`
    (our default).
    """
    job_config = get_job_config()
    metadata: dict = {}

    if job_config.get("adapter_type") == "lora":
        lora_config = job_config.get("lora_config") or {}
        if "lora_alpha" in lora_config:
            metadata["lora_alpha"] = int(lora_config["lora_alpha"])
        if "use_rslora" in lora_config:
            metadata["use_rslora"] = bool(lora_config["use_rslora"])

    return metadata

class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Perform allreduce operation out of place
        input_tmp = input.clone()
        allreduce(input_tmp)
        # Return the all-reduced tensor
        return input_tmp

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output_tmp = grad_output.clone()
        # Perform allreduce operation in place
        allreduce(grad_output_tmp)
        # Return the all-reduced gradient
        return grad_output_tmp

allreduce_op = _AllReduce.apply
