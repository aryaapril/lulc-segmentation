"""Shared configuration, preprocessing and data-access helpers for the Sentinel-1 LULC project.

The dataset notebook, the training notebook and the Streamlit prototype all import this module,
so training and inference use exactly the same SAR preprocessing, class scheme and model loading.
"""
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import odc.stac
import planetary_computer
import pystac_client
from odc.geo.geobox import GeoBox
from pyproj import Transformer
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_VERSION = "v4"
DATA_DIR = PROJECT_ROOT / "data" / DATASET_VERSION
MODEL_DIR = PROJECT_ROOT / "models" / DATASET_VERSION
FIGURE_DIR = PROJECT_ROOT / "figures" / DATASET_VERSION
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"   # the model used by the prototype

# ---------------------------------------------------------------- data source
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-rtc"   # radiometrically terrain-corrected gamma0, linear power, 10 m, UTM
RESOLUTION = 10                    # metres per pixel
# Sentinel-1B failed in Dec 2021, so recent months only have ~5 acquisitions over a site.
# Capping every monthly composite at the same number of acquisition days keeps the speckle
# level of training composites (2021, S1A+S1B) comparable to inference composites (S1A only).
MAX_SCENES_PER_MONTH = 5
LEE_WINDOW = 5
# Assumed equivalent number of looks of one acquisition in the Lee filter. A tuning choice, not a measured value
# (the nominal ENL of Sentinel-1 IW GRDH is ~4.4; a higher value filters less). Fixed because the models were
# trained with it.
LEE_ENL = 10.0

# Fixed dB ranges mapped to [0, 1]. A fixed scale (instead of per-image percentiles) keeps the
# absolute backscatter level, which is what separates water (dark) from urban (bright).
DB_RANGES = {
    "vv": (-25.0, 5.0),
    "vh": (-32.0, -5.0),
    "ratio": (0.0, 20.0),          # VV - VH in dB (cross-pol ratio, sensitive to volume scattering)
    # Std. dev. (dB) of the speckle-filtered backscatter across the month's acquisitions. Mixes look
    # directions (asc/desc), so it captures orientation-dependent scattering (buildings) as well as
    # temporal change (crops, wind on water).
    "vv_std": (0.0, 5.0),
    "vh_std": (0.0, 5.0),
}
CHANNELS = ["vv", "vh", "ratio", "vv_std", "vh_std"]

# ---------------------------------------------------------------- classes
IGNORE_INDEX = 255
CLASS_NAMES = ["Forest", "Grassland", "Cropland", "Urban", "Water", "Bare"]
# Natural hues separated by lightness (dark forest < mid grassland < bright crops < light grey bare) so that every pair
# stays distinguishable with colour blindness (OKLab ΔE >= 6.7 under simulated protan/deutan/tritan vision, >= 16 with
# normal vision). Urban is a deep red: brighter reds match Forest for red-green colour-blind viewers.
CLASS_COLORS = ["#1d7036", "#6cb04a", "#fbc531", "#c92a2a", "#2680e3", "#d4d4d4"]
NUM_CLASSES = len(CLASS_NAMES)

# ESA WorldCover -> training classes. Bare / sparse vegetation is class 5 since dataset v4 (deserts, quarries,
# beaches; v3 ignored it and had to call deserts "Grassland"). Snow/ice is ignored: permanent snow is rare, and
# seasonal snow would contradict the annual label.
WORLDCOVER_TO_CLASS = {
    10: 0,              # tree cover
    95: 0,              # mangroves
    20: 1, 30: 1, 90: 1, 100: 1,   # shrubland, grassland, herbaceous wetland, moss (permanent low vegetation)
    40: 2,              # cropland
    50: 3,              # built-up
    80: 4,              # permanent water
    60: 5,              # bare / sparse vegetation
    70: IGNORE_INDEX, 0: IGNORE_INDEX,
}
# Esri / Impact Observatory 10 m annual LULC (io-lulc-annual-v02) -> same classes.
# Only used as the independent reference map in the prototype, never for training.
ESRI_TO_CLASS = {
    2: 0,               # trees
    4: 1, 11: 1,        # flooded vegetation, rangeland
    5: 2,               # crops
    7: 3,               # built area
    1: 4,               # water
    8: 5,               # bare ground
    9: IGNORE_INDEX, 10: IGNORE_INDEX, 0: IGNORE_INDEX,   # snow/ice, clouds, no data
}
ESRI_YEARS = (2017, 2023)

# WorldCover exists for 2020 (v100) and 2021 (v200) only. Imagery of 2021 is labelled with the
# full v200 map; imagery of every other year only where both maps agree (stable pixels, which
# also drops most v100 errors, since v200 is the more accurate product). This assumes that land cover
# changes slowly, so stable pixels remain valid labels for neighbouring years.
LABEL_YEAR = 2021


def make_lut(mapping):
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut


WORLDCOVER_LUT = make_lut(WORLDCOVER_TO_CLASS)
ESRI_LUT = make_lut(ESRI_TO_CLASS)


# ---------------------------------------------------------------- SAR preprocessing
def lee_filter(img, size=LEE_WINDOW, enl=LEE_ENL):
    """Lee speckle filter on linear-power backscatter (input must not contain NaNs)."""
    mean = uniform_filter(img, size)
    sq_mean = uniform_filter(img * img, size)
    var = np.maximum(sq_mean - mean * mean, 0.0)
    cu2 = 1.0 / enl
    weight = np.clip((var - mean * mean * cu2) / (var * (1.0 + cu2) + 1e-12), 0.0, 1.0)
    return mean + weight * (img - mean)


def scale_db(db, band):
    lo, hi = DB_RANGES[band]
    return np.clip((db - lo) / (hi - lo), 0.0, 1.0)


MIN_OBSERVATIONS = 2   # pixels need >= 2 acquisitions in the month (for the median and the std)


def filtered_db_stack(stack):
    """(T, H, W) linear gamma0 -> (T, H, W) Lee-filtered dB, NaN where an acquisition has no data."""
    out = np.full(stack.shape, np.nan, dtype=np.float32)
    for t, arr in enumerate(np.asarray(stack, dtype=np.float32)):
        ok = np.isfinite(arr) & (arr > 0)
        if not ok.any():
            continue
        arr = np.where(ok, arr, np.median(arr[ok]))   # fill gaps so the filter does not spread NaNs
        out[t] = np.where(ok, 10.0 * np.log10(np.maximum(lee_filter(arr), 1e-6)), np.nan)
    return out


def preprocess_s1(vv_stack, vh_stack):
    """Monthly stacks of linear gamma0 (T, H, W) -> (C, H, W) float32 network input and (H, W) validity mask.

    Per acquisition: mask no-data -> Lee speckle filter (linear) -> 10*log10 dB.
    Per pixel over the month: median (composite) and std (variability) -> VV-VH ratio
    -> fixed-range scaling to [0, 1].
    """
    vv_db, vh_db = filtered_db_stack(vv_stack), filtered_db_stack(vh_stack)
    n_obs = np.minimum(np.isfinite(vv_db).sum(0), np.isfinite(vh_db).sum(0))
    valid = n_obs >= MIN_OBSERVATIONS

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN pixels -> NaN, masked below
        channels = {"vv": np.nanmedian(vv_db, 0), "vh": np.nanmedian(vh_db, 0),
                    "vv_std": np.nanstd(vv_db, 0), "vh_std": np.nanstd(vh_db, 0)}
    channels["ratio"] = channels["vv"] - channels["vh"]

    image = np.stack([scale_db(channels[b], b) for b in CHANNELS], axis=0)
    image[:, ~valid] = 0.0
    return np.nan_to_num(image).astype(np.float32), valid


# ---------------------------------------------------------------- geometry
def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def geobox_from_lonlat_bbox(bbox):
    """[min_lon, min_lat, max_lon, max_lat] -> 10 m UTM GeoBox and the lon/lat bbox for STAC search."""
    lon_c, lat_c = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    epsg = utm_epsg(lat_c, lon_c)
    bounds = Transformer.from_crs(4326, epsg, always_xy=True).transform_bounds(*bbox)
    gbox = GeoBox.from_bbox(bounds, crs=f"epsg:{epsg}", resolution=RESOLUTION, tight=True)
    return gbox, list(bbox)


def geobox_around(lat, lon, size_px):
    """Square size_px x size_px GeoBox at 10 m in the local UTM zone, centred on (lat, lon)."""
    epsg = utm_epsg(lat, lon)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = size_px * RESOLUTION / 2
    bounds = (x - half, y - half, x + half, y + half)
    gbox = GeoBox.from_bbox(bounds, crs=f"epsg:{epsg}", resolution=RESOLUTION, tight=True)
    ll_bbox = Transformer.from_crs(epsg, 4326, always_xy=True).transform_bounds(*bounds)
    return gbox, list(ll_bbox)


# ---------------------------------------------------------------- data access
def open_catalog():
    # Anonymous access works; setting PC_SDK_SUBSCRIPTION_KEY only raises rate limits.
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def month_range(year, month):
    start = date(year, month, 1)
    end = date(year + (month == 12), month % 12 + 1, 1)
    return f"{start.isoformat()}/{end.isoformat()}"


def select_acquisitions(items, max_days=MAX_SCENES_PER_MONTH):
    """Group STAC items by acquisition day and keep up to max_days days spread over the month."""
    items = [it for it in items if {"VV", "VH"} <= set(it.properties.get("sar:polarizations", []))]
    by_day = {}
    for it in sorted(items, key=lambda it: it.datetime):
        by_day.setdefault(it.datetime.date(), []).append(it)
    days = sorted(by_day)
    if len(days) > max_days:
        days = [days[i] for i in np.linspace(0, len(days) - 1, max_days).round().astype(int)]
    return [it for d in days for it in by_day[d]], len(days)


def load_s1_month(catalog, gbox, ll_bbox, year, month, max_days=MAX_SCENES_PER_MONTH):
    """Linear gamma0 VV/VH acquisitions of one calendar month on the given GeoBox.

    Returns (vv_stack, vh_stack, n_days) with stacks of shape (days, H, W), NaN = no data,
    or None if no acquisitions exist. Feed the stacks to preprocess_s1().
    """
    items = catalog.search(collections=[S1_COLLECTION], bbox=ll_bbox,
                           datetime=month_range(year, month)).item_collection()
    items, n_days = select_acquisitions(list(items), max_days)
    if not items:
        return None
    ds = odc.stac.load(items, bands=["vv", "vh"], geobox=gbox, groupby="solar_day",
                       resampling="bilinear", pool=4)
    ds = ds.where(ds > 0)                       # nodata (-32768) and zeros -> NaN
    return ds.vv.values.astype(np.float32), ds.vh.values.astype(np.float32), n_days


def load_categorical(catalog, collection, band, gbox, ll_bbox, datetime, lut=None):
    items = catalog.search(collections=[collection], bbox=ll_bbox, datetime=datetime).item_collection()
    if len(items) == 0:
        return None
    ds = odc.stac.load(items, bands=[band], geobox=gbox, groupby="solar_day", resampling="nearest")
    raw = ds[band].isel(time=0).values.astype(np.uint8)
    return raw if lut is None else lut[raw]


def load_worldcover_raw(catalog, gbox, ll_bbox, year):
    """Raw ESA WorldCover codes (2020 = v100, 2021 = v200); 0 = no data."""
    return load_categorical(catalog, "esa-worldcover", "map", gbox, ll_bbox, f"{year}-01-01/{year}-12-31")


def load_worldcover_labels(catalog, gbox, ll_bbox):
    """ESA WorldCover 2021 (v200) remapped to the training classes."""
    return WORLDCOVER_LUT[load_worldcover_raw(catalog, gbox, ll_bbox, 2021)]


def labels_from_raw(raw_masks, year=LABEL_YEAR, lut=WORLDCOVER_LUT):
    """(2, H, W) raw WorldCover codes [2021, 2020] -> class labels for imagery of the given year.

    year == 2021: full WorldCover 2021 map. Any other year: only pixels where 2020 and 2021 agree
    after remapping; the rest become IGNORE_INDEX (see LABEL_YEAR).
    """
    label = lut[raw_masks[0]]
    if year != LABEL_YEAR:
        label = np.where(lut[raw_masks[1]] == label, label, IGNORE_INDEX).astype(np.uint8)
    return label


def majority_filter(label, size=5):
    """Mode filter over size x size windows, ignoring unlabelled pixels; unlabelled pixels stay unlabelled.

    Matches the label granularity to what ~20 m SAR can resolve (single street trees or gardens vanish).
    """
    votes = np.stack([uniform_filter((label == c).astype(np.float32), size) for c in range(NUM_CLASSES)])
    out = votes.argmax(0).astype(np.uint8)
    out[label == IGNORE_INDEX] = IGNORE_INDEX
    return out


def load_esri_reference(catalog, gbox, ll_bbox, year):
    """Esri 10 m annual LULC remapped to the training classes (independent reference map)."""
    year = int(np.clip(year, *ESRI_YEARS))
    ref = load_categorical(catalog, "io-lulc-annual-v02", "data", gbox, ll_bbox,
                           f"{year}-01-01/{year}-12-31", ESRI_LUT)
    return ref, year


# ---------------------------------------------------------------- statistics
def class_percentages(pred):
    """Percentage of valid (non-ignored) pixels per class. Pixels are equal-area (10 m UTM)."""
    valid = pred != IGNORE_INDEX
    n = valid.sum()
    if n == 0:
        return np.full(NUM_CLASSES, np.nan)
    return np.array([(pred[valid] == c).sum() / n * 100.0 for c in range(NUM_CLASSES)])


def confusion_matrix(pred, target, num_classes=NUM_CLASSES):
    """Rows = target, columns = prediction; pixels ignored in either map are skipped."""
    keep = (pred != IGNORE_INDEX) & (target != IGNORE_INDEX)
    idx = target[keep].astype(np.int64) * num_classes + pred[keep].astype(np.int64)
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    oa = tp.sum() / max(cm.sum(), 1)
    return {"OA": oa, "mIoU": iou.mean(), "IoU": iou, "F1": f1}


# ---------------------------------------------------------------- model
ARCHITECTURES = ("Unet", "DeepLabV3Plus", "UnetPlusPlus")


def build_model(arch, encoder="resnet34", in_channels=len(CHANNELS), encoder_weights=None, classes=NUM_CLASSES):
    import segmentation_models_pytorch as smp
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}, expected one of {ARCHITECTURES}")
    return getattr(smp, arch)(encoder_name=encoder, encoder_weights=encoder_weights,
                              in_channels=in_channels, classes=classes)


def with_temperature(model, temperature):
    """Divides the logits by a temperature fitted on the validation set (calibrated softmax, same argmax)."""
    import torch

    class Calibrated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x) / temperature

    return Calibrated()


def load_checkpoint(path, device):
    """Rebuild a model from a checkpoint written by the training notebooks.

    5-class checkpoints of dataset v3 still load: their classes are the first five of CLASS_NAMES, so they simply
    never predict Bare. A fitted "temperature" in the config is applied to the logits.
    """
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False) if not isinstance(path, dict) else path
    if "members" in ckpt:                     # ensemble checkpoint: {"config", "members": [member checkpoints]}
        built = [load_checkpoint(m, device) for m in ckpt["members"]]
        model = build_ensemble([m for m, _ in built], [c for _, c in built])
        return model.to(device).eval(), ckpt["config"]
    if "config" not in ckpt:
        raise ValueError(f"{path} is a bare state_dict from the old pipeline; retrain with the training notebook")
    cfg = ckpt["config"]
    names = cfg.get("class_names", CLASS_NAMES)
    if names != CLASS_NAMES[:len(names)]:
        raise ValueError(f"{path} predicts {names}, but this code uses {CLASS_NAMES}; retrain the model")
    model = build_model(cfg["arch"], cfg["encoder"], n_input_channels(cfg), classes=len(names))
    model.load_state_dict(ckpt["state_dict"])
    if cfg.get("temperature"):
        model = with_temperature(model, cfg["temperature"])
    return model.to(device).eval(), cfg


def channel_indices(channels):
    return [CHANNELS.index(c) for c in channels]


# ---------------------------------------------------------------- temporal context
# A context model sees the previous, current and next month (3 x channels). Crops and grassland
# differ mainly in how they change over the season, and a short window damps single-month
# flicker. Where a neighbouring month is missing or invalid, the current month is repeated
# (the same rule in training and in the prototype).
def context_months(cfg):
    """Months on each side of the current month the model sees (config value True = 1, False = 0)."""
    return int(cfg.get("context") or 0)


def n_input_channels(cfg):
    return len(cfg["channels"]) * (2 * context_months(cfg) + 1)


def model_input_window(cfg, cur, neighbours=None):
    """(C, H, W) monthly images (all CHANNELS) -> network input for this checkpoint's config.

    neighbours: {offset: (image, valid)} for offsets -k..-1, 1..k (missing offsets may be left out). The
    window is filled from the centre outwards: a missing month, or an invalid pixel in it, is replaced by
    the next month closer to the current one. Months are concatenated in time order (-k ... +k).
    """
    idx = channel_indices(cfg["channels"])
    k = context_months(cfg)
    neighbours = neighbours or {}
    window = {0: cur}
    for d in range(1, k + 1):
        for off in (-d, d):
            base = window[off - 1 if off > 0 else off + 1]
            img, valid = neighbours.get(off, (None, None))
            if img is None:
                img = base
            elif valid is not None:
                img = np.where(valid[None], img, base)
            window[off] = img
    return np.concatenate([window[off][idx] for off in range(-k, k + 1)], axis=0)


def model_input(cfg, cur, prev=None, nxt=None, prev_valid=None, next_valid=None):
    """Same as model_input_window() for models with at most +-1 month of context."""
    return model_input_window(cfg, cur, {-1: (prev, prev_valid), 1: (nxt, next_valid)})


def build_ensemble(members, configs):
    """Averages the softmax of several models. Members may use different context windows: the ensemble
    takes the widest window and gives every member its own centred slice. Returns log-probabilities, so
    softmax(output) is the averaged probability (works with predict_tta and argmax)."""
    import torch

    k_max = max(context_months(c) for c in configs)
    n_ch = len(configs[0]["channels"])

    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.members = torch.nn.ModuleList(members)
            self.slices = [slice((k_max - context_months(c)) * n_ch, (k_max + context_months(c) + 1) * n_ch)
                           for c in configs]

        def forward(self, x):
            prob = sum(torch.softmax(m(x[:, s]).float(), 1) for m, s in zip(self.members, self.slices))
            return torch.log(prob / len(self.members) + 1e-8)

    return Ensemble()


def ensemble_config(name, configs):
    return {"name": name, "arch": "Ensemble", "encoder": " + ".join(c["name"] for c in configs),
            "members": [c["name"] for c in configs], "channels": configs[0]["channels"],
            "context": max(context_months(c) for c in configs), "class_names": CLASS_NAMES,
            **{k: configs[0][k] for k in ("db_ranges", "max_scenes_per_month", "dataset", "label_rule", "train")
               if k in configs[0]}}


def predict_tta(model, x):
    """Softmax averaged over the 4 flips of a (B, C, H, W) batch (test-time augmentation)."""
    import torch
    prob = 0
    for dims in ((), (-1,), (-2,), (-2, -1)):
        xi = torch.flip(x, dims) if dims else x
        p = torch.softmax(model(xi).float(), 1)
        prob = prob + (torch.flip(p, dims) if dims else p)
    return prob / 4
