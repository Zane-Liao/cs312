import json
from pathlib import Path

import modal

from modal_utils import USER_VOLUME_MOUNTS, user_volume

app = modal.App("cs312-wandb-state-cleaner")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface-hub",
        "matplotlib",
        "modal[api-proxy-support]>=1.5.3",
        "numpy",
        "requests",
        "sentencepiece",
        "torch==2.6.0",
        "triton==3.2.0",
        "tqdm",
        "wandb==0.17.3",
    )
    .add_local_dir(
        ".",
        remote_path="/root",
        copy=True,
        ignore=[
            ".venv",
            ".git",
            "__pycache__",
            "*.pyc",
            "data_cache",
            "outputs",
            "checkpoints",
        ],
    )
)


def remove_wandb_ids(obj):
    """
    Recursively remove only W&B run ID fields from JSON data.
    Returns True if anything was changed.
    """
    changed = False

    if isinstance(obj, dict):
        for key in list(obj.keys()):
            normalized = key.lower().replace("-", "_")

            # 最明确的情况
            if normalized == "wandb_run_id":
                print(f"    removing JSON key: {key}")
                del obj[key]
                changed = True
                continue

            # 如果结构类似:
            # "wandb": {"run_id": "..."}
            if normalized == "wandb" and isinstance(obj[key], dict):
                wandb_state = obj[key]

                for wandb_key in list(wandb_state.keys()):
                    wandb_normalized = wandb_key.lower().replace("-", "_")

                    if wandb_normalized in {"run_id", "wandb_run_id"}:
                        print(f"    removing JSON key: {key}.{wandb_key}")
                        del wandb_state[wandb_key]
                        changed = True

                if remove_wandb_ids(wandb_state):
                    changed = True

                continue

            if remove_wandb_ids(obj[key]):
                changed = True

    elif isinstance(obj, list):
        for item in obj:
            if remove_wandb_ids(item):
                changed = True

    return changed


@app.function(
    image=image,
    volumes=USER_VOLUME_MOUNTS,
    timeout=10 * 60,
)
def clean():
    root = Path("/root/data/ckpts")

    if not root.exists():
        print(f"Checkpoint root does not exist: {root}")
        return

    deleted_wandb_files = []
    modified_json_files = []

    print(f"Scanning checkpoint tree: {root}")

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        # --------------------------------------------------
        # 1. 专门的 W&B state 文件
        # --------------------------------------------------

        name_lower = path.name.lower()

        if "wandb" in name_lower:
            print(f"Deleting dedicated W&B state file: {path}")
            path.unlink()
            deleted_wandb_files.append(path)
            continue

        # --------------------------------------------------
        # 2. JSON metadata 中保存的 wandb_run_id
        # --------------------------------------------------

        if path.suffix.lower() == ".json":
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                # 不是正常 JSON 就跳过
                continue

            if remove_wandb_ids(data):
                print(f"Updating metadata: {path}")

                with path.open("w", encoding="utf-8") as f:
                    json.dump(
                        data,
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                    f.write("\n")

                modified_json_files.append(path)

    # 持久化 Volume 修改
    user_volume.commit()

    print()
    print("===== CLEANUP SUMMARY =====")
    print(f"Deleted W&B state files: {len(deleted_wandb_files)}")
    print(f"Modified JSON files:     {len(modified_json_files)}")

    for path in deleted_wandb_files:
        print(f"  deleted:  {path}")

    for path in modified_json_files:
        print(f"  modified: {path}")

    if not deleted_wandb_files and not modified_json_files:
        print()
        print("WARNING: No persisted W&B run IDs were found.")

    print()
    print("Model/checkpoint directories were NOT deleted.")


@app.local_entrypoint()
def main():
    clean.remote()