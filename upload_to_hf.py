"""Publish the model weights and the patch dataset on the Hugging Face Hub (public).

    python upload_to_hf.py <hf-username> <github-repo-url>                  # weights first, then the dataset
    python upload_to_hf.py <hf-username> <github-repo-url> --skip-dataset

The dataset is packed into one uncompressed tar per split and region (data/v4_shards/<split>/<region>.tar), because
the Hub handles a few hundred large files much better than 230,000 small ones. Packing and uploading are resumable:
existing shards are not rebuilt and shards already on the Hub are skipped. Splits are uploaded smallest first, so the
evaluation sets are online early.
"""
import sys
import tarfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "v4"
SHARDS = ROOT / "data" / "v4_shards"
SPLIT_ORDER = ["holdout_arid", "holdout_de", "ood", "test", "val", "train"]

MODEL_CARD = """---
license: cc-by-4.0
tags: [sentinel-1, sar, land-cover, semantic-segmentation, pytorch]
---
# Sentinel-1 land cover U-Net (Unet_ctx2, dataset v4)

U-Net with a ResNet-34 encoder that maps six land cover classes (Forest, Grassland, Cropland, Urban, Water, Bare)
from monthly **Sentinel-1** composites only (VV, VH, VV-VH, VV std, VH std), using the target month and two months on
each side (25 input channels). Confidence is temperature-calibrated (T stored in the checkpoint config).

`best_model.pth` contains the weights and the full config. Load it with `lulc_common.load_checkpoint()` from the
code repository: {code_url}

Test set 2021 (61 regions, spatial block split): OA 78.4 %, mIoU 65.7 % (95 % CI 63.2-67.8), measured against
ESA WorldCover 2021. Labels: ESA WorldCover (CC BY 4.0). Sentinel-1 data: Copernicus, via Microsoft Planetary Computer.
"""

DATASET_CARD = """---
license: cc-by-4.0
tags: [sentinel-1, sar, land-cover, semantic-segmentation, esa-worldcover]
size_categories: [100K<n<1M]
---
# Sentinel-1 monthly land cover patches (dataset v4)

114,676 patches of 256 x 256 px (10 m, local UTM) from 64 regions, one per region, month and year
(2020-2023 for the 49 European regions, 2020-2021 for 12 dry/tropical regions, and 2020-2025 for three regions never
used for training). Built with `src/01_build_dataset.ipynb` of the code repository: {code_url}

* `patches_meta.csv`: one row per patch (region, split, year, month, location, file names).
* `shards/<split>/<region>.tar`: per patch `<stem>_img.npy` (float16, 5 x 256 x 256: VV, VH, VV-VH, VV std,
  VH std, monthly Lee-filtered dB composites scaled to [0, 1]) and `<stem>_mask.npy` (uint8, 2 x 256 x 256: raw ESA
  WorldCover codes 2021 and 2020; 0 = no data). Extract all shards of a split into `data/v4/patches/<split>/`.
* Splits: train / val / test by 5.12 km blocks inside the training regions; holdout_de (Rhine-Neckar), ood
  (Valencia), holdout_arid (Cholistan) are whole regions never used for training.

Sources and licences: Sentinel-1 RTC (Copernicus Sentinel data, processed by Microsoft Planetary Computer);
labels derived from ESA WorldCover 2020 v100 and 2021 v200 (Zanaga et al., CC BY 4.0).
"""


def pack(split, regions):
    """One tar per region of this split; returns the shard paths."""
    files = sorted((DATA / "patches" / split).glob("*.npy"))
    by_region = {}
    for f in files:
        region = next(r for r in regions if f.name.startswith(r + "_"))
        by_region.setdefault(region, []).append(f)
    out = []
    for region, fs in sorted(by_region.items()):
        shard = SHARDS / split / f"{region}.tar"
        if not shard.exists():
            shard.parent.mkdir(parents=True, exist_ok=True)
            tmp = shard.with_suffix(".tar.part")
            with tarfile.open(tmp, "w") as tar:
                for f in fs:
                    tar.add(f, arcname=f.name)
            tmp.rename(shard)
            print(f"  packed {shard.relative_to(ROOT)} ({len(fs)} files, {shard.stat().st_size / 1e9:.2f} GB)")
        out.append(shard)
    return out


def main():
    user, code_url = sys.argv[1], sys.argv[2]
    api = HfApi()

    model_repo = f"{user}/s1-lulc-unet-ctx2"
    api.create_repo(model_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=MODEL_CARD.format(code_url=code_url).encode(), path_in_repo="README.md",
                    repo_id=model_repo, repo_type="model")
    api.upload_file(path_or_fileobj=str(ROOT / "models" / "best_model.pth"), path_in_repo="best_model.pth",
                    repo_id=model_repo, repo_type="model")
    print("weights:", f"https://huggingface.co/{model_repo}")
    if "--skip-dataset" in sys.argv:
        return

    data_repo = f"{user}/s1-lulc-patches-v4"
    api.create_repo(data_repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=DATASET_CARD.format(code_url=code_url).encode(), path_in_repo="README.md",
                    repo_id=data_repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(DATA / "patches_meta.csv"), path_in_repo="patches_meta.csv",
                    repo_id=data_repo, repo_type="dataset")
    print("dataset:", f"https://huggingface.co/datasets/{data_repo}")
    regions = sorted(pd.read_csv(DATA / "patches_meta.csv", usecols=["region"])["region"].unique(), key=len,
                     reverse=True)                     # longest first, so a prefix match is unambiguous
    online = set(api.list_repo_files(data_repo, repo_type="dataset"))
    for split in SPLIT_ORDER:
        print(f"{split}: packing")
        for shard in pack(split, regions):
            target = f"shards/{split}/{shard.name}"
            if target in online:
                continue
            api.upload_file(path_or_fileobj=str(shard), path_in_repo=target, repo_id=data_repo,
                            repo_type="dataset")
            print(f"  uploaded {target}", flush=True)
    print("done:", f"https://huggingface.co/datasets/{data_repo}")


if __name__ == "__main__":
    main()
