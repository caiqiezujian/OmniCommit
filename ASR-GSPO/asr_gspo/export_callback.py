"""Complete every synchronous HF save without interrupting a successful training run."""
import json
import os
from pathlib import Path
import traceback


def complete_checkpoint(output_dir):
    from finalize_export import atomic_write, finalize
    source = os.environ.get("ASR_GSPO_SFT_MODEL")
    merged = Path(str(output_dir) + "-merged")
    try:
        if not source:
            raise ValueError("Missing ASR_GSPO_SFT_MODEL: exact SFT parent is required")
        return finalize(Path(source), merged)
    except Exception as exc:
        # Native model+optimizer checkpoint remains valid for resume. Report
        # failure explicitly; do not silently declare this model Web-ready.
        status = dict(ready=False, error=str(exc), sft_parent=source)
        if merged.is_dir():
            atomic_write(merged / "omnicommit_export_status.json", json.dumps(status, ensure_ascii=False).encode("utf-8"))
        print(f"[export ERROR] {merged}: {exc}; native checkpoint retained; run finalize_export.py after repair", flush=True)
        traceback.print_exc()
        return status


def register_export_callback():
    from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map

    class OmniCommitExport(MegatronCallback):
        def on_train_begin(self):
            from .metrics import snapshots
            self.previous_metrics = snapshots(os.environ.get("ASR_GSPO_LOG_DIR"))

        def on_log(self, logs):
            # Pinned Swift creates its TensorBoard writer on the last rank.
            if int(os.environ.get("RANK", "0")) == int(os.environ.get("WORLD_SIZE", "1")) - 1:
                from .metrics import snapshots, update_logs
                if not hasattr(self, "previous_metrics"):
                    self.previous_metrics = {}
                update_logs(self.previous_metrics, snapshots(os.environ.get("ASR_GSPO_LOG_DIR")), logs)

        def on_save(self, output_dir):
            if int(os.environ.get("RANK", "0")) == 0:
                if getattr(self.args, "async_save", False):
                    raise ValueError("OmniCommit export callback requires --async_save false")
                complete_checkpoint(output_dir)

    megatron_callbacks_map["omnicommit_export"] = OmniCommitExport
