import shutil
from pathlib import Path

import modal

from download_data import download_default_data


# ============================================================
# Modal config
# ============================================================

app = modal.App("cs312-data-setup")

DATA_VOLUME_NAME = "cs312-data"
MODAL_ENVIRONMENT = "main"

VOLUME_MOUNT_PATH = Path("/data")

#
# /datasets/dclm_9p6m_ctx1024/
#     manifest.json
#     train/
#     val/
#
DATASET_DIR = (
    VOLUME_MOUNT_PATH
    / "datasets"
    / "dclm_9p6m_ctx1024"
)


# ============================================================
# Volume
# ============================================================

volume = modal.Volume.from_name(
    DATA_VOLUME_NAME,
    create_if_missing=True,
    environment_name=MODAL_ENVIRONMENT,
    version=1,
)


# ============================================================
# Image
# ============================================================

image = (
    modal.Image.debian_slim(python_version="3.10")
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


# ============================================================
# Helpers
# ============================================================

def valid_dataset(path: Path) -> bool:
    """
    Check whether a complete DCLM dataset already exists.
    """

    required = (
        path / "manifest.json",
        path / "train" / "metadata.json",
        path / "val" / "metadata.json",
    )

    return all(p.is_file() for p in required)


def legacy_dataset_exists() -> bool:
    """
    Detect the old layout:

    /data/
        manifest.json
        train/
        val/
    """

    required = (
        VOLUME_MOUNT_PATH / "manifest.json",
        VOLUME_MOUNT_PATH / "train" / "metadata.json",
        VOLUME_MOUNT_PATH / "val" / "metadata.json",
    )

    return all(p.is_file() for p in required)


def migrate_legacy_dataset():
    """
    Convert:

        /data/manifest.json
        /data/train
        /data/val

    into:

        /data/datasets/dclm_9p6m_ctx1024/
    """

    print()
    print("=" * 60)
    print("LEGACY DATASET DETECTED")
    print("=" * 60)

    print("Migrating existing dataset instead of downloading again.")
    print()

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    for name in (
        "manifest.json",
        "train",
        "val",
    ):
        src = VOLUME_MOUNT_PATH / name
        dst = DATASET_DIR / name

        print(f"Moving:")
        print(f"  {src}")
        print(f"  -> {dst}")

        shutil.move(
            str(src),
            str(dst),
        )

    print()
    print("Legacy dataset migration completed.")


# ============================================================
# Main Modal function
# ============================================================

@app.function(
    image=image,
    volumes={
        str(VOLUME_MOUNT_PATH): volume,
    },
    timeout=60 * 60 * 6,
    cpu=2,
    memory=4096,
)
def setup_dataset(force: bool = False):

    print("=" * 70)
    print("CS312 DATASET SETUP")
    print("=" * 70)

    print()
    print(f"Volume:")
    print(f"  {DATA_VOLUME_NAME}")

    print()
    print(f"Volume mount:")
    print(f"  {VOLUME_MOUNT_PATH}")

    print()
    print(f"Expected dataset directory:")
    print(f"  {DATASET_DIR}")

    print()

    # ========================================================
    # Case 1
    # Dataset already exists in the correct location.
    # ========================================================

    if valid_dataset(DATASET_DIR) and not force:
        print("Dataset already exists in the correct location.")
        print("Nothing needs to be downloaded or migrated.")

        print()
        print(f"Dataset directory:")
        print(f"  {DATASET_DIR}")

        return {
            "status": "already_exists",
            "data_dir": str(DATASET_DIR),
        }

    # ========================================================
    # Case 2
    # Old dataset layout exists.
    #
    # Do NOT download the 18GB again.
    # ========================================================

    if (
        not force
        and not DATASET_DIR.exists()
        and legacy_dataset_exists()
    ):
        migrate_legacy_dataset()

        volume.commit()

        if not valid_dataset(DATASET_DIR):
            raise RuntimeError(
                "Migration completed, but the resulting dataset "
                "failed validation."
            )

        print()
        print("=" * 70)
        print("SUCCESS")
        print("=" * 70)

        print("Existing data was migrated successfully.")
        print()
        print(f"Dataset directory:")
        print(f"  {DATASET_DIR}")

        return {
            "status": "migrated",
            "data_dir": str(DATASET_DIR),
        }

    # ========================================================
    # Case 3
    # Empty/new Volume.
    #
    # Download directly into the final correct location.
    # ========================================================

    print("No usable existing dataset found.")
    print("Downloading dataset directly into final location...")
    print()

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = download_default_data(
        output_dir=DATASET_DIR,
        force=force,
    )

    volume.commit()

    # ========================================================
    # Validate downloaded data
    # ========================================================

    if not valid_dataset(DATASET_DIR):
        raise RuntimeError(
            f"Dataset download finished but validation failed: "
            f"{DATASET_DIR}"
        )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print("Dataset successfully downloaded.")

    print()
    print(f"Data directory:")
    print(f"  {DATASET_DIR}")

    if isinstance(result, dict):
        if "repo_id" in result:
            print(f"Repository:")
            print(f"  {result['repo_id']}")

        if "revision" in result:
            print(f"Revision:")
            print(f"  {result['revision']}")

    # ========================================================
    # Environment diagnostics
    # ========================================================

    print()
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)

    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.version.cuda}")

    try:
        import triton

        print(f"triton: {triton.__version__}")
        print(f"triton path: {triton.__file__}")

    except Exception as exc:
        print(f"TRITON ERROR: {exc!r}")

    return {
        "status": "downloaded",
        "data_dir": str(DATASET_DIR),
        "download_result": result,
    }


# ============================================================
# Local entrypoint
# ============================================================

@app.local_entrypoint()
def main(force: bool = False):
    result = setup_dataset.remote(force=force)

    print()
    print("Result:")
    print(result)