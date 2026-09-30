
import modal

from download_data import download_default_data

app = modal.App("cs312-data-downloader")

DATA_VOLUME_NAME = "cs312-data"
MODAL_ENVIRONMENT = "main"

DATA_MOUNT_PATH = "/data/datasets/dclm_9p6m_ctx1024"

volume = modal.Volume.from_name(
    DATA_VOLUME_NAME,
    create_if_missing=True,
    environment_name=MODAL_ENVIRONMENT,
)

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


@app.function(
    image=image,
    volumes={DATA_MOUNT_PATH: volume},
    timeout=60 * 60 * 6,
    cpu=2,
    memory=4096,
)
def download_dataset(force: bool = False):
    result = download_default_data(
        output_dir=DATA_MOUNT_PATH,
        force=force,
    )

    volume.commit()

    print("Dataset successfully downloaded.")
    print(f"Data directory: {result['data_dir']}")
    print(f"Repository: {result['repo_id']}")
    print(f"Revision: {result['revision']}")

    return result


@app.local_entrypoint()
def main(force: bool = False):
    download_dataset.remote(force=force)
