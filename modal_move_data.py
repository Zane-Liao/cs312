
import shutil
from pathlib import Path

import modal

app = modal.App("cs312-data-migration")

volume = modal.Volume.from_name(
    "cs312-data",
    environment_name="main",
    version=1,
)

ROOT = Path("/data")
DEST = ROOT / "datasets" / "dclm_9p6m_ctx1024"


@app.function(
    volumes={"/data": volume},
    timeout=3600,
)
def move_data():
    if DEST.exists():
        raise RuntimeError(f"Destination already exists: {DEST}")

    required = [
        ROOT / "manifest.json",
        ROOT / "train" / "metadata.json",
        ROOT / "val" / "metadata.json",
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    DEST.mkdir(parents=True)

    for name in ("manifest.json", "train", "val"):
        shutil.move(str(ROOT / name), str(DEST / name))

    volume.commit()

    print("Migration completed!")
    print(f"Dataset directory: {DEST}")


@app.local_entrypoint()
def main():
    move_data.remote()
