# Sentinel-1 LULC Semantic Segmentation

Land Use / Land Cover segmentation from **Sentinel-1 SAR only** (VV, VH), plus a Streamlit prototype that
produces 12 monthly LULC maps for any location, with class shares and month-to-month and total changes in
percentage points.

Classes: **Forest**, **Grassland** (incl. shrubland, herbaceous wetland), **Cropland**, **Urban**, **Water**,
**Bare** (bare ground and sparse vegetation, e.g. deserts, quarries).

## Links

| What | Where |
|---|---|
| Code (training + prototype) | https://github.com/aryaapril/lulc-segmentation |
| Model weights | https://huggingface.co/aryashinde/s1-lulc-unet-ctx2 (`best_model.pth`) |
| Dataset (114,676 patches, 90 GB) | https://huggingface.co/datasets/aryashinde/s1-lulc-patches-v4 |
| Demo video | VIDEO_LINK |

To run the app, download `best_model.pth` into `models/`:

```bash
mkdir -p models
curl -L -o models/best_model.pth https://huggingface.co/aryashinde/s1-lulc-unet-ctx2/resolve/main/best_model.pth
```

The dataset can also be rebuilt from public sources with `src/01_build_dataset.ipynb`. `upload_to_hf.py` is the
script that published the weights and the dataset.

## Repository layout

| Path | Content |
|---|---|
| `src/lulc_common.py` | Shared config and preprocessing (used by both notebooks **and** the app, so training and inference are identical) |
| `src/01_build_dataset.ipynb` | Downloads Sentinel-1 RTC + ESA WorldCover, preprocesses, builds a spatially split patch dataset |
| `src/02_train_evaluate.ipynb` | Model selection on dataset v3 (5 classes, Europe): U-Net / DeepLabV3+ variants, temporal context, test, generalisation, seasonal and change-error analysis |
| `src/03_improve.ipynb` | Dataset v3: ±2-month context model and ensembles; selected `Unet_ctx2` |
| `src/04_global_model.ipynb` | Final model: `Unet_ctx2` retrained on dataset v4 (6 classes, + 12 non-European regions), temperature-calibrated confidence, compared with the v3 model on all test sets; writes `models/best_model.pth` only if it loses ≤ 1 mIoU point in Europe (the final v4 model was copied there by hand: −1.05 points in Europe, +22 points on the added regions). In this notebook "outside Europe" means the 12 added dry/tropical regions, which include Monegros (Spain) |
| `app.py` | Streamlit prototype |
| `models/best_model.pth` | Trained weights + config of the final model (written by notebook 04, used by the app) |
| `models/v3/`, `figures/v3/` | Trained variants, plots and result tables of dataset v3 (notebooks 02, 03) |
| `models/v4/`, `figures/v4/` | Final model, plots and result tables of dataset v4 (notebook 04) |
| `data/` | Generated dataset (not in the repo; see *Dataset*) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # install the CUDA build of torch first if you train on a GPU
```

**Accounts / API keys:** none required. Data comes from the
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC API, which allows anonymous
access. Optionally set `PC_SDK_SUBSCRIPTION_KEY` for higher rate limits. The ImageNet encoder weights are
downloaded automatically from the Hugging Face Hub on first training.

## Reproduce

Run the notebooks from inside `src/`:

1. `01_build_dataset.ipynb` downloads 64 regions (49 European × 2020–2023, 12 dry/tropical × 2020–2021, three generalisation regions also 2024–2025), about 3–4 h on DataLab from scratch, mostly downloading. It is resumable: finished region-years are skipped, so going from v3 to v4 (`mv data/v3 data/v4`) only downloaded the 28 new region-years (18 min). A region-year that fails (e.g. a network error) is listed at the end; re-running the cell fetches only that. It writes `data/v4/patches/…` and `data/v4/patches_meta.csv` (114,676 patches, about 90 GB).
2. `02_train_evaluate.ipynb` and `03_improve.ipynb` are the model-selection experiments on dataset v3 (about 4–5 h and 1.5–2 h on one A6000). Their outputs are kept in `models/v3/` and `figures/v3/`.
3. `04_global_model.ipynb` takes about 3 h on one A6000 (16 min loading, 134 min training, ~15 min calibration and evaluation): it loads dataset v4 into RAM (90 GB), trains `Unet_ctx2` with 6 classes, calibrates its confidence, evaluates it against the v3 model with TTA, and writes `models/v4/` and `figures/v4/` (and `models/best_model.pth`, see the table above).

## Run the prototype

```bash
streamlit run app.py
```

Input: latitude/longitude (typed, picked from example locations, or clicked on a satellite map), start year and
month, and area size. The app:

1. builds a monthly Sentinel-1 RTC median composite (≤ 5 acquisition days) for each of the 12 months (plus
   the context months before and after that the model uses: two on each side for the selected model) on a 10 m UTM grid centred on the point, and
   preprocesses it exactly like the training data;
2. runs the model on each month (optionally averaged over 4 flips, test-time augmentation) and keeps the
   per-pixel confidence (temperature-calibrated softmax probability);
3. reports the class share per month (% of valid pixels), the month-to-month change and the total change
   (first → last month) in percentage points, each computed on the pixels valid in both months compared, plus the
   share of pixels that changed class;
4. compares the 12-month majority map with the independent **Esri 10 m Annual LULC** map (not used for training).

The results are shown in these tabs:

| Tab | Content |
|---|---|
| Map viewer | Overlay on satellite or OSM tiles: land cover per month (optionally hiding pixels below 50 % confidence), 12-month majority, model confidence, change since the first month, change frequency, Sentinel-1 VV |
| Monthly change | Animation (play/pause, month slider, speed) of the monthly maps or of the pixels that changed since the previous month, with the class shares or pp changes per month; GIF download; all 12 maps (PNG) |
| Class statistics | Stacked class shares, area table, month-to-month and total change in pp |
| Change analysis | Sankey diagram and transition matrix (first → last month), how often pixels changed class, gross change compared with the model's mean apparent-change rate on the 2021 test areas (where little real change is expected; an average, not a significance test) |
| Validation | Agreement with Esri LULC: maps, disagreement map, IoU/F1 per class, confusion matrix, agreement per month |
| Model | Model card, test scores (all test blocks, the added dry/tropical regions, the unseen desert region), scores on every evaluation set, training curves, confusion matrices and calibration plot from `figures/v4/`, limitations |

Exports: statistics (CSV), a GeoTIFF with the 12 monthly maps plus the majority map (UTM, 10 m), and a JSON run report.

To reach the app from a remote JupyterHub (e.g. DataLab), a tunnel such as `cloudflared tunnel --url http://localhost:8501` can be used.

## Method summary

| Step | Choice |
|---|---|
| SAR input | `sentinel-1-rtc` (radiometrically terrain-corrected γ⁰, calibrated, 10 m), VV + VH |
| Speckle filter | Lee 5×5 on linear power, per acquisition |
| Compositing | ≤ 5 acquisition days spread over the calendar month; per-pixel median (dB) and std (dB) |
| Channels | VV, VH, VV−VH, VV std, VH std; fixed dB ranges → [0, 1]. Context models: ±1 month (15 channels) or ±2 months (25 channels, selected model); a missing neighbour month is filled with the next month closer to the current one |
| Imagery years | European regions 2020–2023, dry/tropical regions 2020–2021 (both WorldCover years); also 2024 and 2025 for the three generalisation regions |
| Labels | ESA WorldCover, pixel-aligned on the same grid. 2021 imagery: full WorldCover 2021; other years: pixels where WorldCover 2020 and 2021 agree. Bare / sparse vegetation is a class since v4; snow/ice is ignored |
| Regions | 61 training regions: 49 in Central/Northern Europe (37 around cities + 12 rural) and 12 in dry and tropical climates (India, Egypt, Morocco, Spain, USA, Iran, Saudi Arabia, Argentina, Niger, China, Brazil). Never trained on: Rhine-Neckar (DE), Valencia (ES, out-of-distribution) and Cholistan (PK, arid) |
| Split | 5.12 km blocks → train/val/test; all months and years of a block in the same split; no overlapping patches (no leakage) |
| Model | U-Net with ImageNet ResNet-34/50, weighted CE + Dice, AdamW, one-cycle LR (40 epochs, one random year per location-month per epoch), bf16, gradient clipping. Selected on v3 validation: `Unet_ctx2` (ResNet-34, ±2 months), over the ±1-month model and 2–4-model ensembles (an ensemble within 0.2 pp mIoU loses to the simpler model); retrained on v4 |
| Calibration | Temperature scaling of the logits (one parameter), fitted on half of the validation patches without TTA; the expected calibration error is reported on the same pixels. Calibration is only established for data like the validation set (Europe + training regions) |
| Evaluation | Headline: 2021 imagery vs. full WorldCover 2021; 95 % CIs by bootstrap over blocks; test blocks of the European and of the added dry/tropical regions, and three unseen regions; v4 compared with the v3 model (also on the five v3 classes); measured false-change rate |
