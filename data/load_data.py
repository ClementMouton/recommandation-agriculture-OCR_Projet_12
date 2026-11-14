
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "OC-20250526T142605Z-1-001.zip"
)


def load_crop_yield_data(
    archive_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Charge le dataset agricole depuis les archives ZIP imbriquées."""

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Archive introuvable : {archive_path}"
        )

    with ZipFile(archive_path) as outer_zip:
        inner_archive = (
            "OC/Agriculture Crop Yield.zip"
        )

        with outer_zip.open(inner_archive) as archive:
            archive_bytes = BytesIO(archive.read())

        with ZipFile(archive_bytes) as inner_zip:
            with inner_zip.open("crop_yield.csv") as csv_file:
                df = pd.read_csv(csv_file)

    return df
