import base64
import calendar
import io
import json
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from affine import Affine
from dateutil.relativedelta import relativedelta
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:                       # the app still works without the interactive maps
    folium = None

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import lulc_common as lc

st.set_page_config(page_title="Sentinel-1 Land Cover Monitor", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.5rem;}
.legend-item {display: inline-flex; align-items: center; margin: 0 14px 6px 0; font-size: 0.875rem;}
.legend-swatch {width: 12px; height: 12px; border-radius: 2px; margin-right: 6px;
                box-shadow: inset 0 0 0 1px rgba(128, 128, 128, .55);}
.gradient {height: 10px; border-radius: 2px; margin: 4px 0;}
</style>
""", unsafe_allow_html=True)

IGN = lc.IGNORE_INDEX
NO_DATA_COLOR = "#4a4a4a"                 # darker than the light grey of Bare
UNCHANGED_RGBA = [40, 40, 40, 90]         # unchanged pixels in change views: dark, translucent (Bare is light grey)
LMAP = ListedColormap(lc.CLASS_COLORS)
LMAP.set_bad(NO_DATA_COLOR)
LEGEND = [Patch(facecolor=c, edgecolor="#808080", linewidth=0.5, label=n)
          for c, n in zip(lc.CLASS_COLORS, lc.CLASS_NAMES)] + [Patch(color=NO_DATA_COLOR, label="no data")]
BAR_OUTLINE = dict(color="rgba(128, 128, 128, 0.6)", width=1)   # keeps the pale Bare bar visible on a white page
AREA_OPTIONS = {"2.56 km × 2.56 km": 256, "5.12 km × 5.12 km": 512, "10.24 km × 10.24 km": 1024}
ZOOM = {256: 15, 512: 14, 1024: 13}
CUSTOM = "Custom (type or click on the map)"
PLACES = {
    "Heidelberg, DE (held-out region)": (49.4100, 8.6900),
    "Mannheim, DE (held-out region)": (49.4875, 8.4660),
    "Valencia, ES (out-of-distribution)": (39.4699, -0.3763),
    "Tesla Gigafactory Grünheide, DE (forest to industry 2020–22)": (52.3960, 13.7960),
    "Berlin Brandenburg Airport, DE": (52.3667, 13.5033),
    "Leipzig, DE": (51.3397, 12.3731),
    "Cholistan desert edge, PK (unseen arid region)": (28.0417, 70.0000),
    "Amsterdam, NL": (52.3676, 4.9041),
    "Paris, FR": (48.8566, 2.3522),
}
BASEMAPS = {
    "Satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                  "Esri, Maxar, Earthstar Geographics"),
    "OpenStreetMap": ("OpenStreetMap", None),
    "Light": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/"
              "{z}/{y}/{x}", "Esri"),
}
LAYERS = ["Land cover", "12-month majority", "Model confidence", "Change since first month",
          "Change frequency (year)", "Sentinel-1 VV"]
MONTHLY_LAYERS = {"Land cover", "Model confidence", "Change since first month", "Sentinel-1 VV"}
CLASS_RGBA = np.zeros((256, 4), np.uint8)
for _i, _c in enumerate(lc.CLASS_COLORS):
    CLASS_RGBA[_i] = [*(np.array(to_rgb(_c)) * 255).astype(np.uint8), 255]
UNCHANGED = 254                           # marker value in change maps
LOW_CONFIDENCE = 0.5                      # "Hide pixels below ..." threshold in the map viewer
DARK = (st.context.theme.type or "light") == "dark"
FG = "#e6e6e6" if DARK else "#262730"     # text colour of the matplotlib figures


# ---------------------------------------------------------------- location state
st.session_state.setdefault("lat", 49.4100)
st.session_state.setdefault("lon", 8.6900)
st.session_state.setdefault("place", list(PLACES)[0])
if "pending_point" in st.session_state:            # a click on the location map (set before the widgets exist)
    st.session_state.lat, st.session_state.lon = st.session_state.pop("pending_point")
    st.session_state.place = CUSTOM


def apply_place():
    if st.session_state.place in PLACES:
        st.session_state.lat, st.session_state.lon = PLACES[st.session_state.place]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Location")
    st.selectbox("Example locations", [CUSTOM] + list(PLACES), key="place", on_change=apply_place)
    lat = st.number_input("Latitude", min_value=-80.0, max_value=84.0, format="%.4f", step=0.001, key="lat")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, format="%.4f", step=0.001, key="lon")
    area_label = st.selectbox("Area (centred on the point)", list(AREA_OPTIONS), index=1)
    size_px = AREA_OPTIONS[area_label]

    st.subheader("Period")
    this_year = date.today().year
    c1, c2 = st.columns(2)
    start_year = c1.selectbox("Start year", list(range(2017, this_year + 1)), index=2025 - 2017)
    start_month = c2.selectbox("Start month", list(range(1, 13)), index=0,
                               format_func=lambda m: calendar.month_abbr[m])
    end = date(start_year, start_month, 1) + relativedelta(months=11)
    st.caption(f"12 months: {calendar.month_abbr[start_month]} {start_year} – {end:%b %Y}")

    with st.expander("Inference options"):
        tta = st.toggle("Test-time augmentation", value=torch.cuda.is_available(),
                        help="Averages the model output over 4 flipped copies of the input (as in the evaluation). "
                             "Slightly more accurate, 4x slower; recommended on a GPU.")
        smooth = st.toggle("3-month majority filter", value=False,
                           help="Replaces each pixel by the majority class of the previous, current and next "
                                "month. Suppresses single-month flicker caused by seasonal SAR effects. "
                                "Off = raw per-month model output.")
    run = st.button("Run 12-month analysis", type="primary", width="stretch")


# ---------------------------------------------------------------- cached work
@st.cache_resource
def get_model():
    if not lc.BEST_MODEL_PATH.exists():
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = lc.load_checkpoint(lc.BEST_MODEL_PATH, device)
    return model, cfg, device


@st.cache_data(show_spinner=False, max_entries=128)
def fetch_month(lat, lon, size_px, year, month):
    catalog = lc.open_catalog()
    gbox, ll_bbox = lc.geobox_around(lat, lon, size_px)
    res = lc.load_s1_month(catalog, gbox, ll_bbox, year, month)
    if res is None:
        return None
    vv, vh, n_days = res
    image, valid = lc.preprocess_s1(vv, vh)
    return image, valid, n_days


@st.cache_data(show_spinner=False, max_entries=32)
def fetch_reference(lat, lon, size_px, year):
    catalog = lc.open_catalog()
    gbox, ll_bbox = lc.geobox_around(lat, lon, size_px)
    return lc.load_esri_reference(catalog, gbox, ll_bbox, year)


@st.cache_data(show_spinner=False)
def load_evaluation(model_name, fig_dir):
    """Test-set results and false-change rate of this model, written by the training notebooks (if present)."""
    out = {}
    res_path, chg_path = fig_dir / "results.csv", fig_dir / "spurious_change.csv"
    if res_path.exists():
        res = pd.read_csv(res_path)
        out["results"] = res[res["model"].isin([model_name, f"{model_name} + TTA"])]
    if chg_path.exists():
        chg = pd.read_csv(chg_path)
        out["change"] = chg[chg["model"].isin([model_name, f"{model_name} + TTA"])]
    return out


def predict(model, device, x, tta=False):
    """Class map and confidence (softmax probability of the chosen class)."""
    x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = lc.predict_tta(model, x) if tta else torch.softmax(model(x).float(), 1)
    conf, cls = prob[0].max(0)
    return cls.cpu().numpy().astype(np.uint8), conf.cpu().numpy().astype(np.float16)


# ---------------------------------------------------------------- analysis helpers
def class_counts(stack):
    return np.stack([(stack == c).sum(axis=0) for c in range(lc.NUM_CLASSES)])


def temporal_majority(preds):
    """Majority over a 3-month window; keeps the current class unless two months agree on another."""
    out = []
    for i, cur in enumerate(preds):
        if cur is None:
            out.append(None)
            continue
        window = np.stack([p for p in preds[max(0, i - 1):i + 2] if p is not None])
        counts = class_counts(window)
        m = np.where(counts.max(axis=0) >= 2, counts.argmax(axis=0), cur).astype(np.uint8)
        m[cur == IGN] = IGN
        out.append(m)
    return out


def consensus(preds):
    stack = np.stack([p for p in preds if p is not None])
    counts = class_counts(stack)
    out = counts.argmax(axis=0).astype(np.uint8)
    out[counts.sum(axis=0) == 0] = IGN
    return out


def change_frequency(preds):
    """Number of class changes per pixel between consecutive available months (IGN where never valid)."""
    avail = [p for p in preds if p is not None]
    n = np.zeros(avail[0].shape, np.uint8)
    for a, b in zip(avail[:-1], avail[1:]):
        n += (a != b) & (a != IGN) & (b != IGN)
    n[np.all(np.stack(avail) == IGN, axis=0)] = IGN
    return n


def change_map(a, b):
    """New class where a pixel changed from map a to map b, UNCHANGED where not, IGN where either is invalid."""
    return np.where((a == IGN) | (b == IGN), IGN, np.where(a == b, UNCHANGED, b)).astype(np.uint8)


def paired_shares(a, b):
    """Class shares (%) of maps a and b over the pixels valid in both, so their difference is real class change
    and not a change in data coverage."""
    both = (a != IGN) & (b != IGN)
    return lc.class_percentages(np.where(both, a, IGN)), lc.class_percentages(np.where(both, b, IGN))


def share_change(a, b):
    """Share(b) - share(a) in pp over the pixels valid in both maps."""
    sa, sb = paired_shares(a, b)
    return sb - sa


def transition_matrix(a, b):
    """Rows = class in map a, columns = class in map b, in % of the pixels valid in both."""
    cm = lc.confusion_matrix(b, a)
    return cm / max(cm.sum(), 1) * 100


# ---------------------------------------------------------------- rendering helpers
def show_map(ax, pred, title):
    ax.imshow(np.ma.masked_equal(pred, IGN), cmap=LMAP, vmin=0, vmax=lc.NUM_CLASSES - 1, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def mpl_figure(*args, **kwargs):
    """Matplotlib figure styled for the active Streamlit theme (transparent background)."""
    with plt.rc_context({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG}):
        return plt.subplots(*args, **kwargs)


def fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def map_legend(fig, fontsize=11):
    leg = fig.legend(handles=LEGEND, loc="lower center", ncol=len(LEGEND), fontsize=fontsize, frameon=False)
    for t in leg.get_texts():
        t.set_color(FG)


def memo(state, key, fn):
    """Per-run cache in session state (figures are rendered once, not on every widget interaction)."""
    cache = state.setdefault("memo", {})
    key = (key, DARK)
    if key not in cache:
        cache[key] = fn()
    return cache[key]


def legend_html(items):
    return "".join(f'<span class="legend-item"><span class="legend-swatch" style="background:{c}"></span>{n}</span>'
                   for c, n in items)


def gradient_html(cmap, lo, hi):
    stops = ", ".join(matplotlib.colors.to_hex(plt.get_cmap(cmap)(v)) for v in np.linspace(0, 1, 8))
    return (f'<div class="gradient" style="background: linear-gradient(to right, {stops})"></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:.8rem"><span>{lo}</span>'
            f'<span>{hi}</span></div>')


def warp_to_mercator(arr, geo, nodata):
    """UTM array -> Web Mercator array (nearest) and its [[south, west], [north, east]] bounds for Leaflet."""
    src_t = Affine(*geo["transform"])
    h, w = arr.shape
    dst_t, dw, dh = calculate_default_transform(geo["crs"], "EPSG:3857", w, h, *array_bounds(h, w, src_t))
    out = np.full((dh, dw), nodata, dtype=arr.dtype)
    reproject(arr, out, src_transform=src_t, src_crs=geo["crs"], dst_transform=dst_t, dst_crs="EPSG:3857",
              resampling=Resampling.nearest, src_nodata=nodata, dst_nodata=nodata)
    west, north = dst_t * (0, 0)
    east, south = dst_t * (dw, dh)
    to_ll = Transformer.from_crs(3857, 4326, always_xy=True)
    (w_lon, e_lon), (s_lat, n_lat) = to_ll.transform([west, east], [south, north])
    return out, [[s_lat, w_lon], [n_lat, e_lon]]


def colorize(layer, arr):
    """Layer array -> RGBA image."""
    if layer in ("Land cover", "12-month majority"):
        return CLASS_RGBA[arr]
    if layer == "Change since first month":       # changed pixels in the colour of their new class
        rgba = CLASS_RGBA[arr].copy()
        rgba[arr == UNCHANGED] = UNCHANGED_RGBA
        return rgba
    if layer == "Change frequency (year)":
        rgba = (plt.get_cmap("YlOrRd")(np.clip(arr / 4, 0, 1) * 0.8 + 0.2) * 255).astype(np.uint8)
        rgba[(arr == 0) | (arr == IGN), 3] = 0
        return rgba
    if layer == "Model confidence":
        rgba = (plt.get_cmap("viridis")(np.clip((arr - 0.2) / 0.8, 0, 1)) * 255).astype(np.uint8)
    else:                                          # Sentinel-1 VV, scaled dB in [0, 1]
        rgba = (plt.get_cmap("gray")(np.clip(arr, 0, 1)) * 255).astype(np.uint8)
    rgba[arr < 0, 3] = 0
    return rgba


def png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def png_data_uri(rgba):
    return "data:image/png;base64," + base64.b64encode(png_bytes(rgba)).decode()


def base_map(center, zoom, basemap="Satellite"):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    for name, (tiles, attr) in BASEMAPS.items():
        folium.TileLayer(tiles, attr=attr, name=name, show=name == basemap).add_to(m)
    return m


def footprint(m, ll_bbox, color="#ffcc00", pad=0.0):
    w, s, e, n = ll_bbox
    folium.Rectangle([[s, w], [n, e]], color=color, weight=2, fill=False, tooltip="Analysis area").add_to(m)
    dx, dy = (e - w) * pad, (n - s) * pad
    m.fit_bounds([[s - dy, w - dx], [n + dy, e + dx]])


def geotiff_bytes(preds, cons, labels, geo, model_name):
    stack = np.stack([p if p is not None else np.full_like(cons, IGN) for p in preds] + [cons])
    h, w = cons.shape
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=h, width=w, count=len(stack), dtype="uint8", crs=geo["crs"],
                     transform=Affine(*geo["transform"]), nodata=IGN, compress="deflate") as ds:
            ds.write(stack)
            for i, name in enumerate(labels + ["12-month majority"], start=1):
                ds.set_band_description(i, name)
            ds.write_colormap(1, {i: tuple(CLASS_RGBA[i]) for i in range(lc.NUM_CLASSES)})
            ds.update_tags(classes=json.dumps(dict(enumerate(lc.CLASS_NAMES))), model=model_name,
                           source="Sentinel-1 RTC (Microsoft Planetary Computer)")
        return mf.read()


def animation_frames(view, preds, labels, available, pct, gross):
    """(title, RGBA image, bar values) per available month for the monthly animation."""
    frames = []
    for n, i in enumerate(available):
        if view == "Land cover":
            rgba = CLASS_RGBA[preds[i]]
            title = f"{labels[i]}"
            values = pct.iloc[i].to_numpy()
        else:
            prev = available[n - 1] if n else i
            cmap_ = change_map(preds[prev], preds[i])
            rgba = CLASS_RGBA[cmap_].copy()
            rgba[cmap_ == UNCHANGED] = UNCHANGED_RGBA
            changed = (cmap_ < lc.NUM_CLASSES).sum() / max((cmap_ != IGN).sum(), 1) * 100
            title = (f"{labels[i]}: {changed:.1f}% of pixels changed class since {labels[prev]}" if n
                     else f"{labels[i]}: first month")
            values = share_change(preds[prev], preds[i])
        frames.append((title, rgba, values))
    return frames


def animation_figure(frames, view, frame_ms):
    bar_title = "Class share (%)" if view == "Land cover" else "Change vs. previous month (pp)"
    top = max(np.nanmax(np.abs(v)) for _, _, v in frames)
    x_range = [0, top * 1.3] if view == "Land cover" else [-max(top * 1.3, 0.5), max(top * 1.3, 0.5)]

    def traces(rgba, values):
        return [go.Image(source=png_data_uri(rgba), hoverinfo="skip"),
                go.Bar(x=values, y=lc.CLASS_NAMES, orientation="h", marker_color=lc.CLASS_COLORS, marker_line=BAR_OUTLINE,
                       text=[f"{v:+.2f}" if view != "Land cover" else f"{v:.1f}%" for v in values],
                       textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.2f}<extra></extra>")]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], horizontal_spacing=0.08,
                        subplot_titles=("", bar_title))
    for t in traces(frames[0][1], frames[0][2]):
        fig.add_trace(t, row=1, col=1 if isinstance(t, go.Image) else 2)
    fig.frames = [go.Frame(name=title, data=traces(rgba, values), traces=[0, 1],
                           layout=go.Layout(title_text=title)) for title, rgba, values in frames]
    play = dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0), fromcurrent=True)
    fig.update_layout(
        title=dict(text=frames[0][0], x=0, font=dict(size=16)), height=660, margin=dict(l=0, r=10, t=60, b=40),
        showlegend=False,
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.1, xanchor="left", yanchor="top",
                          pad=dict(r=8, t=8), showactive=False,
                          buttons=[dict(label="Play", method="animate", args=[None, play]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, len=0.85, y=-0.1, yanchor="top", pad=dict(t=8),
                      currentvalue=dict(visible=False),
                      steps=[dict(label=title.split(":")[0], method="animate",
                                  args=[[title], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                             for title, _, _ in frames])])
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, scaleanchor="x", row=1, col=1)
    fig.update_xaxes(range=x_range, zeroline=True, row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    return fig


def animation_gif(frames, frame_ms):
    images = []
    for title, rgba, _ in frames:
        bg = Image.new("RGBA", (rgba.shape[1], rgba.shape[0] + 28), (255, 255, 255, 255))
        bg.alpha_composite(Image.fromarray(rgba, "RGBA"), (0, 28))
        ImageDraw.Draw(bg).text((8, 7), title, fill=(20, 20, 20, 255))
        images.append(bg.convert("P", palette=Image.ADAPTIVE))
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=frame_ms, loop=0)
    return buf.getvalue()


def describe_model(cfg):
    k = lc.context_months(cfg)
    window = ("single month" if k == 0 else "previous, current and next month" if k == 1
              else f"{2 * k + 1} months centred on the current one")
    if cfg["arch"] == "Ensemble":
        arch = f"Ensemble of {len(cfg['members'])} U-Nets ({', '.join(cfg['members'])}), softmax averaged"
    else:
        arch = f"{cfg['arch']} with {cfg['encoder']} encoder (ImageNet initialisation)"
    return arch, window


# ---------------------------------------------------------------- header
model, cfg, device = get_model()
st.title("Sentinel-1 Land Cover Monitor")
st.markdown("Monthly land use / land cover maps from Sentinel-1 radar only, with class shares and "
            "percentage-point (pp) changes over 12 months.")
if model is None:
    st.error(f"No trained model at `{lc.BEST_MODEL_PATH}`. Run the notebooks in `src/` first.")
    st.stop()
st.caption(f"Model: {cfg['name']} · input: VV + VH, {describe_model(cfg)[1]} · {device.type.upper()}"
           + (" · test-time augmentation" if tta else ""))

# ---------------------------------------------------------------- location picker
gbox_now, ll_now = lc.geobox_around(lat, lon, size_px)
with st.expander("Pick the location on a map", expanded="run" not in st.session_state):
    if folium is None:
        st.info("Install `folium` and `streamlit-folium` for the interactive map; use the sidebar inputs meanwhile.")
    else:
        m = base_map([lat, lon], ZOOM[size_px] - 2)
        footprint(m, ll_now, pad=0.6)
        folium.CircleMarker([lat, lon], radius=5, color="#ffcc00", fill=True, fill_opacity=1,
                            tooltip=f"{lat:.4f}, {lon:.4f}").add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
        out = st_folium(m, key="picker", height=380, use_container_width=True, returned_objects=["last_clicked"])
        click = (out or {}).get("last_clicked")
        if click:
            pt = (round(float(np.clip(click["lat"], -80, 84)), 4), round(float((click["lng"] + 180) % 360 - 180), 4))
            if pt != st.session_state.get("handled_click"):
                st.session_state.handled_click = pt
                st.session_state.pending_point = pt
                st.rerun()
        st.caption("Click to move the analysis area (yellow box).")

# ---------------------------------------------------------------- run
if run:
    gbox, ll_bbox = gbox_now, ll_now
    geo = {"crs": str(gbox.crs), "transform": tuple(gbox.affine)[:6], "ll_bbox": ll_bbox}
    start = date(start_year, start_month, 1)
    # A context model also needs the months before the first and after the last month.
    pad = lc.context_months(cfg)
    months = [start + relativedelta(months=i) for i in range(-pad, 12 + pad)]
    data = {}
    with st.status("Running the 12-month analysis", expanded=True) as status:
        progress = st.progress(0.0)
        for i, d in enumerate(months):
            progress.progress(i / (len(months) + 1), text=f"{d:%b %Y}: searching Sentinel-1 RTC and compositing")
            if d > date.today():
                data[d] = "in the future"
            else:
                try:
                    data[d] = fetch_month(lat, lon, size_px, d.year, d.month) or "no Sentinel-1 data"
                except Exception as e:
                    data[d] = f"download failed ({type(e).__name__})"
            ctx = " (context month)" if not 0 <= i - pad < 12 else ""
            st.write(f"{d:%b %Y}{ctx}: " + (data[d] if isinstance(data[d], str)
                                            else f"composite of {data[d][2]} acquisition days"))

        progress.progress(len(months) / (len(months) + 1), text="Segmenting")
        results = []
        for d in months[pad:len(months) - pad]:
            empty = {"label": d.strftime("%b %Y"), "date": d, "pred": None, "conf": None, "vv": None,
                     "n_days": 0, "valid_pct": 0.0}
            if isinstance(data[d], str):
                results.append({**empty, "note": data[d]})
                continue
            image, valid, n_days = data[d]
            if not valid.any():               # every pixel has < 2 acquisitions: nothing to composite
                results.append({**empty, "n_days": n_days, "note": f"too few acquisitions ({n_days} day)"})
                continue
            neighbours = {}
            for off in [o for o in range(-pad, pad + 1) if o]:
                other = data.get(d + relativedelta(months=off), "")
                if not isinstance(other, str):
                    neighbours[off] = (other[0], other[1])
            x = lc.model_input_window(cfg, image, neighbours)
            pred, conf = predict(model, device, x, tta)
            pred[~valid] = IGN
            conf[~valid] = -1
            results.append({**empty, "pred": pred, "conf": conf, "vv": np.where(valid, image[0], -1).astype(np.float16),
                            "n_days": n_days, "valid_pct": valid.mean() * 100, "note": ""})

        progress.progress(1.0, text="Loading the independent reference map")
        try:   # the calendar year holding most of the 12 months
            ref, ref_year = fetch_reference(lat, lon, size_px, (start + relativedelta(months=5)).year)
        except Exception as e:
            st.warning(f"Could not load the reference map: {e}")
            ref, ref_year = None, None
        n_ok = sum(r["pred"] is not None for r in results)
        status.update(label=f"Analysis complete: {n_ok}/12 months mapped", state="complete", expanded=False)
    st.session_state["run"] = {"id": time.time(), "params": (lat, lon, start_year, start_month, area_label),
                               "results": results, "ref": ref, "ref_year": ref_year, "geo": geo,
                               "settings": {"tta": tta, "model": cfg["name"]}}

if "run" not in st.session_state:
    st.info("Choose a location (example list, coordinates or map) and a start month, then press "
            "**Run 12-month analysis**.")
    st.stop()

state = st.session_state["run"]
p_lat, p_lon, p_year, p_month, p_area = state["params"]
if state["params"] != (lat, lon, start_year, start_month, area_label):
    st.warning("The inputs changed. The results below are for the previous run; press **Run** to update.")

results = state["results"]
labels = [r["label"] for r in results]
raw_preds = [r["pred"] for r in results]
preds = temporal_majority(raw_preds) if smooth else raw_preds
available = [i for i, p in enumerate(preds) if p is not None]
if not available:
    st.error("No Sentinel-1 data was found for any of the 12 months.")
    st.stop()

# ---------------------------------------------------------------- statistics
pct = pd.DataFrame([lc.class_percentages(p) if p is not None else [np.nan] * lc.NUM_CLASSES for p in preds],
                   index=labels, columns=lc.CLASS_NAMES)
# Changes compare two maps on the pixels valid in both (as the transition matrix and the pixel change rate do).
m2m = pd.DataFrame([share_change(a, b) if a is not None and b is not None else [np.nan] * lc.NUM_CLASSES
                    for a, b in zip([None] + preds[:-1], preds)], index=labels, columns=lc.CLASS_NAMES)
first, last = available[0], available[-1]
first_pct, last_pct = paired_shares(preds[first], preds[last])
total = pd.DataFrame({f"{labels[first]} (%)": first_pct, f"{labels[last]} (%)": last_pct,
                      "Total change (pp)": last_pct - first_pct}, index=lc.CLASS_NAMES)
gross = [np.nan] + [
    ((a != b) & (a != IGN) & (b != IGN)).sum() / max(((a != IGN) & (b != IGN)).sum(), 1) * 100
    if a is not None and b is not None else np.nan
    for a, b in zip(preds[:-1], preds[1:])
]
cons = consensus(raw_preds)
mean_conf = [float(r["conf"][r["conf"] >= 0].mean()) if r["conf"] is not None else np.nan for r in results]
FIG_DIR = lc.PROJECT_ROOT / "figures" / cfg.get("dataset", lc.DATASET_VERSION)   # results of the loaded model
evaluation = load_evaluation(cfg["name"], FIG_DIR)
# Apparent change on the 2021 test areas (same model and TTA setting, unfiltered output): the per-month pixel flip
# rate and, per class, the mean |share(last month) - share(first month)|. The labels are annual, so real change
# within the year is assumed small there; these are averages, not confidence bounds.
noise_floor, noise_pp = None, {}
if "change" in evaluation and not smooth:
    chg = evaluation["change"]
    run_name = f"{cfg['name']} + TTA" if state["settings"]["tta"] else cfg["name"]
    row = chg[(chg["model"] == run_name) & (chg["set"] == "test 2021")].dropna()
    if len(row):
        noise_floor = float(row["pixels changing class per month (%)"].iloc[0])
        noise_pp = {n: float(row[f"{n} |first→last| pp"].iloc[0]) for n in lc.CLASS_NAMES
                    if f"{n} |first→last| pp" in row}
if noise_pp:
    total["Typical model noise (pp)"] = pd.Series(noise_pp)
    total["Above typical noise"] = np.where(total["Total change (pp)"].abs() > total["Typical model noise (pp)"],
                                            "yes", "no")

ref, ref_year = state["ref"], state["ref_year"]
ref_year_matches = ref_year == (date(p_year, p_month, 1) + relativedelta(months=5)).year
ref_metrics = None
if ref is not None:
    ref_cm = lc.confusion_matrix(cons, ref)
    ref_metrics = lc.metrics_from_confusion(ref_cm)
    # Classes absent from both maps would count as IoU 0; average only over classes present in this area.
    present = (ref_cm.sum(axis=0) + ref_cm.sum(axis=1)) > 0
    ref_metrics["present"] = present
    ref_metrics["mIoU"] = ref_metrics["IoU"][present].mean() if present.any() else np.nan
    ref_metrics["mF1"] = ref_metrics["F1"][present].mean() if present.any() else np.nan
monthly_oa = [lc.metrics_from_confusion(lc.confusion_matrix(p, ref))["OA"] * 100 if p is not None and ref is not None
              else np.nan for p in raw_preds]
month_index = pd.to_datetime([r["date"] for r in results])

st.caption(f"Results: {p_lat:.4f}, {p_lon:.4f} · {p_area} · {labels[0]} – {labels[-1]} · 10 m pixels in "
           f"{state['geo']['crs']}" + (" · 3-month majority filter on" if smooth else ""))

# ---------------------------------------------------------------- KPIs
KPI_HEIGHT = 190
mean_pct = pct.mean()
dominant = mean_pct.idxmax()
net = total["Total change (pp)"]
biggest = net.abs().idxmax()
mean_gross = np.nanmean(gross) if len(available) > 1 else np.nan
k = st.columns(5)
k[0].metric("Months mapped", f"{len(available)}/12", f"{sum(r['n_days'] for r in results)} acquisition days",
            delta_color="off", delta_arrow="off", border=True, height=KPI_HEIGHT, chart_type="bar",
            chart_data=[r["n_days"] for r in results],
            help="Months with Sentinel-1 data. Bars: acquisition days used per month.")
k[1].metric("Dominant class", dominant, f"{mean_pct[dominant]:.1f}% of area", delta_color="off",
            delta_arrow="off", border=True, height=KPI_HEIGHT, chart_data=pct[dominant].dropna().round(2).tolist(),
            help="Largest mean share over the 12 months; line = its share per month.")
k[2].metric("Largest net change", biggest,
            f"{net[biggest]:+.2f} pp" + (f" (typical noise {noise_pp[biggest]:.1f})" if biggest in noise_pp else ""),
            delta_color="off", border=True, height=KPI_HEIGHT, chart_data=pct[biggest].dropna().round(2).tolist(),
            help=f"Class with the largest share change from {labels[first]} to {labels[last]} (pixels valid in both "
                 "months); line = its share per month. Noise = the first-to-last change this model reports on "
                 "average on the 2021 test areas, where little real change is expected.")
k[3].metric("Monthly pixel change", f"{mean_gross:.1f}%",
            f"model noise {noise_floor:.1f}%" if noise_floor is not None else None,
            delta_color="off", delta_arrow="off", border=True, height=KPI_HEIGHT, chart_type="bar",
            chart_data=[0 if np.isnan(g) else round(g, 2) for g in gross[1:]],
            help="Mean share of pixels whose class differs from the previous month. On the 2021 test areas, where "
                 "little real change is expected, the model already flips about the 'noise' share; values near it "
                 "are probably not real change. Not shown with the majority filter on (measured without it).")
k[4].metric("Agreement with Esri", f"{ref_metrics['OA'] * 100:.1f}%" if ref_metrics else "–",
            f"Esri LULC {ref_year}" if ref_metrics else None, delta_color="off", delta_arrow="off",
            border=True, height=KPI_HEIGHT, chart_data=[round(v, 1) for v in monthly_oa if not np.isnan(v)] or None,
            help="Share of pixels where the 12-month majority map has the same class as the independent Esri 10 m "
                 "map (itself a model output, not ground truth); line = agreement of each monthly map.")

# Tracked tabs: a widget change inside a tab reruns the app but keeps that tab open.
tabs = st.tabs(["Map viewer", "Monthly change", "Class statistics", "Change analysis", "Validation", "Model"],
               key="active_tab", on_change="rerun")

# ---------------------------------------------------------------- 1. map viewer
with tabs[0]:
    c_ctrl, c_map = st.columns([1, 3], gap="large")
    with c_ctrl:
        layer = st.radio("Layer", LAYERS, index=0)
        month_opts = [labels[i] for i in available]
        month = st.select_slider("Month", month_opts, value=month_opts[0],
                                 disabled=layer not in MONTHLY_LAYERS or len(month_opts) == 1)
        opacity = st.slider("Opacity", 0.0, 1.0, 0.75, 0.05)
        basemap = st.segmented_control("Basemap", list(BASEMAPS), default="Satellite", required=True)
        hide_uncertain = st.toggle(f"Hide pixels below {LOW_CONFIDENCE:.0%} confidence", value=False,
                                   disabled=layer != "Land cover",
                                   help="Leaves pixels transparent where the probability of the predicted class is "
                                        "below 50%; on the validation data such pixels were wrong more often than "
                                        "right.")
        mi = labels.index(month)
        r = results[mi]

        if layer in ("Land cover", "12-month majority", "Change since first month"):
            st.markdown(legend_html(zip(lc.CLASS_COLORS, lc.CLASS_NAMES)), unsafe_allow_html=True)
        if layer == "Change since first month":
            st.caption(f"Coloured pixels changed class between {labels[first]} and {month}; the colour is the new "
                       "class. Dark tint = unchanged.")
        elif layer == "Model confidence":
            st.markdown(gradient_html("viridis", "low (0.2)", "high (1.0)"), unsafe_allow_html=True)
            st.caption("Probability of the predicted class" + (" (calibrated on the validation set: there, of the "
                       "pixels at 80%, about 80% agreed with WorldCover; in unfamiliar regions it can be "
                       "overconfident)" if cfg.get("temperature") else "") + ". Low values mark ambiguous pixels.")
        elif layer == "Change frequency (year)":
            st.markdown(gradient_html("YlOrRd", "1 change", "4 or more"), unsafe_allow_html=True)
            st.caption("How often each pixel switched class between consecutive months. Repeated switching is "
                       "more likely seasonal SAR effects (e.g. crop cycles) or model uncertainty than real "
                       "land-cover change.")
        elif layer == "Sentinel-1 VV":
            st.caption("Monthly median VV backscatter (scaled dB).")
        if layer in MONTHLY_LAYERS:
            st.caption(f"{month}: {r['n_days']} acquisition days · mean confidence {mean_conf[mi] * 100:.0f}%")
        uncertain = (r["conf"] >= 0) & (r["conf"] < LOW_CONFIDENCE) & (preds[mi] != IGN)
        if hide_uncertain and layer == "Land cover":
            st.caption(f"Hidden: {uncertain.sum() / max((preds[mi] != IGN).sum(), 1) * 100:.1f}% of the area. "
                       "The class shares below still count every pixel.")

    if layer == "Land cover":
        arr, nodata = (np.where(uncertain, IGN, preds[mi]) if hide_uncertain else preds[mi]), IGN
    elif layer == "12-month majority":
        arr, nodata = cons, IGN
    elif layer == "Model confidence":
        arr, nodata = r["conf"].astype(np.float32), -1.0
    elif layer == "Change since first month":
        arr, nodata = change_map(preds[first], preds[mi]), IGN
    elif layer == "Change frequency (year)":
        arr, nodata = change_frequency(preds), IGN
    else:
        arr, nodata = r["vv"].astype(np.float32), -1.0

    with c_map:
        if folium is None:
            st.image(png_bytes(colorize(layer, arr)), width="stretch")
        else:
            warped, bounds = warp_to_mercator(arr, state["geo"], nodata)
            m = base_map([p_lat, p_lon], ZOOM[AREA_OPTIONS[p_area]], basemap)
            folium.raster_layers.ImageOverlay(png_data_uri(colorize(layer, warped)), bounds=bounds,
                                              opacity=opacity, name=layer).add_to(m)
            footprint(m, state["geo"]["ll_bbox"], pad=0.05)
            st_folium(m, key="viewer", height=600, use_container_width=True, returned_objects=[], pixelated=True)

        if layer in ("Land cover", "12-month majority"):
            shares = lc.class_percentages(preds[mi] if layer == "Land cover" else arr)
            bar = go.Figure(go.Bar(x=shares, y=lc.CLASS_NAMES, orientation="h", marker_color=lc.CLASS_COLORS,
                                   marker_line=BAR_OUTLINE,
                                   text=[f"{s:.1f}%" for s in shares], textposition="auto"))
            bar.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="% of area",
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(bar, key="share_bar")

# ---------------------------------------------------------------- 2. monthly change (animation)
with tabs[1]:
    c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="bottom")
    view = c1.segmented_control("Show", ["Land cover", "Changes"], default="Land cover", required=True,
                                help="Changes: pixels that switched class since the previous month, in the colour "
                                     "of their new class; unchanged pixels are darkened.")
    speed = c2.select_slider("Speed (seconds per month)", [0.4, 0.7, 1.0, 1.5, 2.0], value=1.0)
    if len(available) < 2:
        st.info("At least two months with data are needed for an animation.")
    else:
        frames = memo(state, ("frames", view, smooth), lambda: animation_frames(view, preds, labels, available,
                                                                                pct, gross))
        st.plotly_chart(animation_figure(frames, view, int(speed * 1000)), key="animation",
                        config={"displayModeBar": False})
        st.markdown(legend_html(list(zip(lc.CLASS_COLORS, lc.CLASS_NAMES))
                                + ([("#282828", "unchanged")] if view == "Changes" else [])),
                    unsafe_allow_html=True)
        with c3:
            st.download_button("Download animation (GIF)",
                               memo(state, ("gif", view, smooth, speed),
                                    lambda: animation_gif(frames, int(speed * 1000))),
                               f"lulc_{'maps' if view == 'Land cover' else 'changes'}.gif", "image/gif")

    def grid_png():
        fig, axes = mpl_figure(3, 4, figsize=(16, 12.8))
        for ax, r, p in zip(axes.ravel(), results, preds):
            if p is None:
                ax.text(0.5, 0.5, r["note"], ha="center", va="center", fontsize=11, wrap=True)
                ax.set_title(r["label"], fontsize=11, fontweight="bold")
                ax.axis("off")
            else:
                show_map(ax, p, f"{r['label']}  ({r['n_days']} acquisition days)")
        map_legend(fig, 12)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig_png(fig)

    with st.expander("All 12 maps"):
        png = memo(state, ("grid", smooth), grid_png)
        st.image(png, width="stretch")
        st.download_button("Download the 12 maps (PNG)", png, "lulc_monthly_maps.png", "image/png")

# ---------------------------------------------------------------- 3. class statistics
with tabs[2]:
    area = go.Figure()
    for name, color in zip(lc.CLASS_NAMES, lc.CLASS_COLORS):
        area.add_trace(go.Scatter(x=month_index, y=pct[name], name=name, stackgroup="one", mode="lines",
                                  line=dict(width=0.5, color=color), fillcolor=color,
                                  hovertemplate="%{y:.2f}%<extra>" + name + "</extra>"))
    area.update_layout(title="Class share per month (% of valid pixels)", height=380, hovermode="x unified",
                       yaxis=dict(range=[0, 100], ticksuffix="%"), margin=dict(t=50, b=10),
                       legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(area, key="area")

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("**Area per class (%)**")
        st.dataframe(pct.assign(**{"Acquisition days": [r["n_days"] for r in results],
                                   "Valid pixels (%)": [r["valid_pct"] for r in results]})
                     .style.format("{:.2f}", na_rep="–").format("{:d}", subset=["Acquisition days"])
                     .background_gradient(cmap="Greens", subset=lc.CLASS_NAMES, axis=None))
        st.markdown("**Month-to-month change (pp, pixels valid in both months)**")
        lim = max(np.nanmax(np.abs(m2m.values)) if m2m.notna().any().any() else 1, 0.5)
        st.dataframe(m2m.assign(**{"Pixels changing class (%)": gross}).style.format("{:+.2f}", na_rep="–")
                     .background_gradient(cmap="RdBu", subset=lc.CLASS_NAMES, vmin=-lim, vmax=lim))
    with c2:
        st.markdown(f"**Total change, {labels[first]} to {labels[last]} (pp)**")
        noise = [noise_pp.get(n, 0) for n in lc.CLASS_NAMES]
        tot = go.Figure(go.Bar(x=net.values, y=lc.CLASS_NAMES, orientation="h", marker_color=lc.CLASS_COLORS,
                               marker_line=BAR_OUTLINE,
                               text=[f"{v:+.2f}" for v in net.values], textposition="outside",
                               error_x=dict(type="data", array=noise, visible=bool(noise_pp), color="gray")))
        span = max((net.abs() + noise).max() * 1.35, 0.5)
        tot.update_layout(height=260, margin=dict(l=0, r=10, t=10, b=0), xaxis=dict(range=[-span, span],
                          zeroline=True, title="pp"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(tot, key="total")
        st.dataframe(total.style.format({total.columns[0]: "{:.2f}", total.columns[1]: "{:.2f}",
                                         total.columns[2]: "{:+.2f}", "Typical model noise (pp)": "{:.2f}"}))
        if noise_pp:
            st.caption("Whiskers / typical model noise: the mean first-to-last change this model reports on the 2021 "
                       "test areas (mostly Europe), where little real change is expected. It is an average, not a "
                       "significance test, and noise is higher in unfamiliar regions (about 2× in the unseen desert "
                       "test region). Changes well above it are more likely real.")
        elif smooth:
            st.caption("Model noise is not shown with the 3-month majority filter (it was measured without it).")
        if first != 0 or last != 11:
            st.caption(f"First/last months without data were skipped: comparing {labels[first]} with "
                       f"{labels[last]}.")
    st.caption("Percentages are computed over valid pixels only (no-data pixels excluded); all pixels have the same "
               "area (10 m × 10 m UTM). Changes compare two months on the pixels valid in both, so data gaps do not "
               "show up as change: month-to-month change = share(month) − share(previous month), total change = "
               "share(last month) − share(first month), in percentage points. Where coverage differs between months "
               "the month-to-month changes therefore need not add up exactly to the total.")

# ---------------------------------------------------------------- 4. change analysis
with tabs[3]:
    tm = transition_matrix(preds[first], preds[last])
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        src, dst, val, col = [], [], [], []
        for i in range(lc.NUM_CLASSES):
            for j in range(lc.NUM_CLASSES):
                if tm[i, j] >= 0.05:
                    rgb = (np.array(to_rgb(lc.CLASS_COLORS[i])) * 255).astype(int)
                    src.append(i)
                    dst.append(lc.NUM_CLASSES + j)
                    val.append(tm[i, j])
                    col.append(f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{0.25 if i == j else 0.75})")
        sankey = go.Figure(go.Sankey(
            textfont=dict(size=13, shadow="none"),
            node=dict(label=[f"{n} ({labels[first][:3]})" for n in lc.CLASS_NAMES]
                      + [f"{n} ({labels[last][:3]})" for n in lc.CLASS_NAMES],
                      color=lc.CLASS_COLORS * 2, pad=18, thickness=16),
            link=dict(source=src, target=dst, value=val, color=col,
                      hovertemplate="%{source.label} to %{target.label}: %{value:.2f}% of area<extra></extra>")))
        sankey.update_layout(title=f"Class transitions, {labels[first]} to {labels[last]}", height=430,
                             margin=dict(t=50, b=10))
        st.plotly_chart(sankey, key="sankey")
    with c2:
        st.metric(f"Area with the same class in {labels[first]} and {labels[last]}", f"{np.trace(tm):.1f}%",
                  border=True)
        st.markdown("**Transitions (% of area)**: rows = first month, columns = last month")
        st.dataframe(pd.DataFrame(tm, index=lc.CLASS_NAMES, columns=lc.CLASS_NAMES).style.format("{:.2f}")
                     .background_gradient(cmap="Purples", axis=None))

    freq = change_frequency(preds)
    n_changes = freq[freq != IGN]
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.markdown("**How stable is each pixel over the year?**")
        dist = pd.Series({"never changed": (n_changes == 0).mean() * 100,
                          "changed once (persistent)": (n_changes == 1).mean() * 100,
                          "changed 2–3 times": ((n_changes >= 2) & (n_changes <= 3)).mean() * 100,
                          "changed 4 or more times (flicker)": (n_changes >= 4).mean() * 100}, name="% of pixels")
        st.dataframe(dist.to_frame().style.format("{:.1f}"))
        st.caption("A real land-cover change (e.g. construction, clear-cut) usually switches a pixel once. "
                   "Pixels that switch back and forth are more likely seasonal SAR effects (crop growth, soil "
                   "moisture, flooding) or model uncertainty than real change. The 3-month majority filter "
                   "suppresses them.")
        if noise_floor is not None:
            st.info(f"On the 2021 test areas, where little real change is expected, this model flips "
                    f"**{noise_floor:.1f}%** of pixels per month (this run: {mean_gross:.1f}%). Month-to-month "
                    "changes of that size are probably model noise.")
        st.caption("Where these pixels are: map viewer, layer *Change frequency (year)*.")
    with c2:
        st.markdown("**Pixels changing class per month (%)**: gross change. It is larger than the net pp change "
                    "because gains and losses of a class cancel in the net figure.")
        gross_fig = go.Figure(go.Bar(x=month_index, y=gross, marker_color="#7e57c2", name="this run",
                                     hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra></extra>"))
        if noise_floor is not None:
            gross_fig.add_hline(y=noise_floor, line_dash="dash", line_color="gray",
                                annotation_text="model noise on the 2021 test areas")
        gross_fig.update_layout(height=300, margin=dict(t=10, b=10), yaxis_title="% of pixels")
        st.plotly_chart(gross_fig, key="gross")

# ---------------------------------------------------------------- 5. validation
with tabs[4]:
    if ref is None:
        st.warning("Reference map unavailable for this location.")
    else:
        cm = lc.confusion_matrix(cons, ref)
        m = ref_metrics
        st.markdown(f"Reference: **Esri / Impact Observatory 10 m Annual LULC {ref_year}**, remapped to the same six "
                    "classes. It was not used for training (the model learned from ESA WorldCover 2020/2021), and "
                    "it is itself a model output (from Sentinel-2), so this is a plausibility check, not ground truth. "
                    "Model map = most frequent class per pixel over the available months. The two label sources draw "
                    "class boundaries differently (e.g. how sparse vegetation must be to count as Bare rather than "
                    "Grassland), so part of the disagreement is definition, not error."
                    + (f" Esri is available for {lc.ESRI_YEARS[0]}–{lc.ESRI_YEARS[1]}; the {ref_year} map is used "
                       f"for this period." if not ref_year_matches else ""))
        v = st.columns(3)
        v[0].metric("Overall agreement", f"{m['OA'] * 100:.1f}%", border=True)
        n_present = int(m["present"].sum())
        v[1].metric("Mean IoU", f"{m['mIoU'] * 100:.1f}%", border=True,
                    help=f"Averaged over the {n_present} classes that occur in the model or reference map here.")
        v[2].metric("Mean F1", f"{m['mF1'] * 100:.1f}%", border=True,
                    help=f"Averaged over the {n_present} classes that occur in the model or reference map here.")

        def ref_png():
            agree = np.where((cons == IGN) | (ref == IGN), IGN, (cons != ref).astype(np.uint8))
            fig, axes = mpl_figure(1, 3, figsize=(18, 6.4))
            show_map(axes[0], cons, "Model: 12-month majority")
            show_map(axes[1], ref, f"Esri LULC {ref_year} (remapped)")
            axes[2].imshow(np.ma.masked_equal(agree, IGN), cmap=ListedColormap(["#cfd8dc", "#e53935"]), vmin=0,
                           vmax=1, interpolation="nearest")
            axes[2].set_title(f"Disagreement (red): {100 - m['OA'] * 100:.1f}%", fontsize=11, fontweight="bold")
            axes[2].axis("off")
            map_legend(fig)
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            return fig_png(fig)
        st.image(memo(state, "ref", ref_png), width="stretch")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Class shares: model vs. reference**")
            cmp = pd.DataFrame({"Model (%)": lc.class_percentages(cons), "Reference (%)": lc.class_percentages(ref)},
                               index=lc.CLASS_NAMES)
            cmp["Difference (pp)"] = cmp["Model (%)"] - cmp["Reference (%)"]
            cmp["IoU (%)"] = np.where(m["present"], m["IoU"] * 100, np.nan)
            cmp["F1 (%)"] = np.where(m["present"], m["F1"] * 100, np.nan)
            st.dataframe(cmp.style.format("{:.2f}").background_gradient(cmap="RdYlGn", subset=["IoU (%)", "F1 (%)"],
                                                                        vmin=0, vmax=100))
            oa_fig = go.Figure(go.Scatter(x=month_index, y=monthly_oa, mode="lines+markers", line_color="#1f6feb",
                                          hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>"))
            oa_fig.add_hline(y=m["OA"] * 100, line_dash="dash", line_color="gray", annotation_text="12-month majority")
            oa_fig.update_layout(title="Agreement of each monthly map with the reference", height=280,
                                 margin=dict(t=50, b=10), yaxis_title="OA (%)")
            st.plotly_chart(oa_fig, key="monthly_oa")
            st.caption("The reference is one annual map, so differences between months come from the model's "
                       "monthly maps, not from the reference.")
        with c2:
            cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
            heat = go.Figure(go.Heatmap(z=cm_pct, x=lc.CLASS_NAMES, y=lc.CLASS_NAMES, colorscale="Blues", zmin=0,
                                        zmax=100, text=np.round(cm_pct, 1), texttemplate="%{text}",
                                        hovertemplate="reference %{y}, model %{x}: %{z:.1f}%<extra></extra>"))
            heat.update_layout(title="Confusion (% of each reference class)", height=420, margin=dict(t=50, b=10),
                               xaxis_title="model", yaxis=dict(title="reference", autorange="reversed"))
            st.plotly_chart(heat, key="confusion")

# ---------------------------------------------------------------- 6. model
with tabs[5]:
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.markdown("**Model card**")
        train = cfg.get("train", {})
        arch, window = describe_model(cfg)
        card = {
            "Name": cfg["name"],
            "Architecture": arch,
            "Input": " + ".join(cfg["channels"]) + f" for the {window}",
            "Input channels": lc.n_input_channels(cfg),
            "Classes": ", ".join(cfg.get("class_names", lc.CLASS_NAMES)),
            "Training labels": cfg.get("label_rule", "ESA WorldCover"),
            "Training imagery years": ", ".join(map(str, train.get("years", []))) or "–",
            "Epochs / batch size": f"{train.get('epochs', '–')} / {train.get('batch_size', '–')}",
            "Composite": f"up to {cfg.get('max_scenes_per_month', lc.MAX_SCENES_PER_MONTH)} acquisition days per "
                         f"month, Lee {lc.LEE_WINDOW}×{lc.LEE_WINDOW} filter, median + std in dB",
            "Dataset": cfg.get("dataset", lc.DATASET_VERSION)
                       + (f", {train['regions']} training regions" if "regions" in train else ""),
            "Confidence": (f"calibrated (temperature {cfg['temperature']:.2f}, validation ECE "
                           f"{cfg['ece']['before'] * 100:.1f}% → {cfg['ece']['after'] * 100:.1f}%)"
                           if cfg.get("temperature") else "raw softmax (not calibrated)"),
        }
        st.markdown("| | |\n|---|---|\n" + "\n".join(f"| **{k}** | {v} |" for k, v in card.items()))
    with c2:
        res = evaluation.get("results")
        if res is None or not len(res):
            st.info(f"No evaluation results found in `{FIG_DIR}`. They are written by the training notebooks.")
        else:
            best = res[res["model"].str.endswith("+ TTA")] if res["model"].str.endswith("+ TTA").any() else res
            best = best.set_index("set")
            st.markdown(f"**Evaluation of {best['model'].iloc[0]}** (spatially independent test blocks and "
                        "regions never used in training)")
            e = st.columns(3)
            for col, (label, s) in zip(e, [("Test mIoU", "test 2021"), ("Test mIoU, added dry/tropical regions", "test 2021 global"),
                                           ("Unseen desert (Cholistan)", "holdout_arid 2021")]):
                if s in best.index:
                    col.metric(label, f"{best.loc[s, 'mIoU (%)']:.1f}%", border=True,
                               help=f"{s}: 95% CI {best.loc[s, 'mIoU 95% CI']}, overall accuracy "
                                    f"{best.loc[s, 'OA (%)']:.1f}%")
            st.dataframe(best[["OA (%)", "OA 95% CI", "mIoU (%)", "mIoU 95% CI"]], width="stretch")
            if "test 2021" in best.index:
                names = [n for n in lc.CLASS_NAMES if f"{n} IoU" in best.columns]
                iou = [best.loc["test 2021", f"{n} IoU"] for n in names]
                iou_fig = go.Figure(go.Bar(x=names, y=iou, marker_color=lc.CLASS_COLORS[:len(names)],
                                           marker_line=BAR_OUTLINE,
                                           text=[f"{v:.1f}" for v in iou], textposition="outside"))
                iou_fig.update_layout(title="Per-class IoU on the test set (%)", height=280, margin=dict(t=50, b=10),
                                      yaxis=dict(range=[0, 105]))
                st.plotly_chart(iou_fig, key="iou")
    for fname, caption in [("training_curves.png", "Training curves"),
                           ("confusion_matrices.png", "Confusion matrices"),
                           ("calibration.png", "Confidence calibration")]:
        path = FIG_DIR / fname
        if path.exists():
            with st.expander(caption):
                st.image(str(path), width="stretch")
    st.markdown("**Limitations**")
    st.markdown(
        "- Sentinel-1 only: radar measures structure and moisture, not colour. Bare and Grassland have the lowest "
        "test IoU (see the chart above); Grassland is confused with Cropland, Urban and Forest, Bare with Urban "
        "and Grassland.\n"
        "- Labels come from ESA WorldCover (itself a model; reported overall accuracy about 75%), so the model "
        "inherits its errors and "
        "its 2020/2021 snapshot. Bare and sparse vegetation (WorldCover) vs. Grassland is a fuzzy boundary in dry "
        "regions.\n"
        "- Trained on 49 European regions plus 12 regions in dry and tropical climates. Snow and ice, rainforest "
        "and tundra are not covered; check the confidence layer there.\n"
        "- Some regions get only 2–3 acquisition days per month (e.g. Riyadh, Niamey, Brasília in the training "
        "data), so their monthly composites are noisier; the days used are shown per month.\n"
        "- Month-to-month changes include seasonal SAR effects; persistent changes (one switch, then stable) are "
        "the trustworthy signal.")

# ---------------------------------------------------------------- export
with st.sidebar:
    st.subheader("Export")
    csv = pd.concat({"area_pct": pct, "m2m_pp": m2m}, axis=1)
    csv[("m2m_pp", "pixels_changing_class_pct")] = gross
    st.download_button("Statistics (CSV)", csv.to_csv().encode(), "lulc_statistics.csv", "text/csv",
                       width="stretch")
    tif = memo(state, ("tif", smooth), lambda: geotiff_bytes(preds, cons, labels, state["geo"], cfg["name"]))
    st.download_button("Maps (GeoTIFF, 13 bands)", tif, "lulc_maps.tif", "image/tiff", width="stretch",
                       help=f"Band 1–12: monthly class maps, band 13: 12-month majority. Values 0–{lc.NUM_CLASSES - 1} = "
                            + ", ".join(f"{i} {n}" for i, n in enumerate(lc.CLASS_NAMES)) + "; 255 = no data.")
    report = {
        "location": {"lat": p_lat, "lon": p_lon, "area": p_area, "crs": state["geo"]["crs"]},
        "period": [labels[0], labels[-1]], "model": cfg["name"], "tta": state["settings"]["tta"],
        "temporal_filter": smooth,
        "months": [{"month": r["label"], "s1_days": r["n_days"], "note": r["note"] or None,
                    "class_pct": None if p is None else dict(zip(lc.CLASS_NAMES, np.round(lc.class_percentages(p), 3)
                                                                 .tolist()))}
                   for r, p in zip(results, preds)],
        "total_change_pp": dict(zip(lc.CLASS_NAMES, np.round(net.values, 3).tolist())),
        "model_noise_pp": noise_pp or None,
        "reference": None if ref is None else {"map": f"Esri 10 m Annual LULC {ref_year}",
                                               "overall_agreement": round(float(ref_metrics["OA"]), 4),
                                               "mIoU": round(float(ref_metrics["mIoU"]), 4)},
    }
    st.download_button("Run report (JSON)", json.dumps(report, indent=2, default=str).encode(), "lulc_report.json",
                       "application/json", width="stretch")
