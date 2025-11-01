"""
DataLens Thermal Studio — Streamlit MVP
---------------------------------------
A Streamlit desktop-like app that replicates core workflows from FLIR Thermal Studio:
- Batch import of thermal images or temperature matrices (CSV/NPY/TIFF)
- Global scale normalization + unified colormap
- Quick analysis (min/max/mean, hot/cold spots)
- Video creator (MP4/GIF) from frames
- Report generator (PDF) from a lightweight template

Author: DataLens.Tools
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.cm as cm
import cv2  # opencv-python
from fpdf import FPDF  # fpdf2

# ------------------------------------------------------------------------------------
# Constants / supported formats
# ------------------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_MATRIX_EXTS = {".csv", ".npy"}

# session keys
SK_FRAMES = "frames"
SK_GLOBAL_MINMAX = "global_minmax"
SK_TEMPLATE = "report_template"


# ------------------------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------------------------
@dataclass
class ThermalFrame:
    name: str
    temp: np.ndarray  # temperature matrix in Celsius (float32)

    @property
    def stats(self) -> dict:
        t = self.temp
        return {
            "min": float(np.nanmin(t)),
            "max": float(np.nanmax(t)),
            "mean": float(np.nanmean(t)),
            "p95": float(np.nanpercentile(t, 95)),
        }


# ------------------------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="DataLens Thermal Studio",
    page_icon="🔥",
    layout="wide",
)

# init session
if SK_FRAMES not in st.session_state:
    st.session_state[SK_FRAMES] = []  # type: List[ThermalFrame]
if SK_GLOBAL_MINMAX not in st.session_state:
    st.session_state[SK_GLOBAL_MINMAX] = None  # (vmin, vmax)
if SK_TEMPLATE not in st.session_state:
    st.session_state[SK_TEMPLATE] = {
        "title": "Thermal Analysis Report",
        "subtitle": "Generated with DataLens.Tools",
        "footer": "© 2025 DataLens.Tools — Confidential — For internal use only",
        "logo": None,
    }

# Sidebar styling
st.markdown(
    """
<style>
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]{
  display:flex; flex-direction:column; height:100%;
}
.sidebar-footer{
  margin-top:auto;
  padding:12px 10px;
  font-size:12px;
  color:#6b7280;
  border-top:1px solid rgba(0,0,0,.06);
}
.sidebar-footer a{ color:inherit; text-decoration:none; font-weight:600; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def _load_unicode_font_safe(pdf: FPDF) -> str:
    """
    Try to register a Unicode font (for local/dev use).
    On Streamlit Cloud this may fail -> we return 'Helvetica' instead of crashing.
    """
    try:
        # typical Linux path; if present we can use it
        base = Path("/usr/share/fonts/truetype/dejavu")
        regular = base / "DejaVuSans.ttf"
        bold = base / "DejaVuSans-Bold.ttf"
        italic = base / "DejaVuSans-Oblique.ttf"

        if regular.exists():
            pdf.add_font("DejaVu", "", str(regular), uni=True)
        if bold.exists():
            pdf.add_font("DejaVu", "B", str(bold), uni=True)
        if italic.exists():
            pdf.add_font("DejaVu", "I", str(italic), uni=True)
        return "DejaVu"
    except Exception:
        # cloud-safe fallback
        return "Helvetica"


def read_temperature_from_image(img: Image.Image, min_c: float, max_c: float) -> np.ndarray:
    """
    Convert a grayscale/pseudo-colored image to a temperature estimate.
    Assumes pixel intensity ∈ [0,255] maps linearly to [min_c, max_c].
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    temp = min_c + (gray / 255.0) * (max_c - min_c)
    return temp.astype(np.float32)


def load_file_as_frame(file, min_c: float, max_c: float) -> Optional[ThermalFrame]:
    name = file.name
    ext = os.path.splitext(name)[1].lower()

    try:
        if ext in SUPPORTED_IMAGE_EXTS:
            img = Image.open(file)
            temp = read_temperature_from_image(img, min_c, max_c)
            return ThermalFrame(name=name, temp=temp)
        elif ext == ".csv":
            df = pd.read_csv(file, header=None)
            return ThermalFrame(name=name, temp=df.values.astype(np.float32))
        elif ext == ".npy":
            temp = np.load(file)
            return ThermalFrame(name=name, temp=temp.astype(np.float32))
        else:
            st.warning(f"Unsupported file type: {ext}")
            return None
    except Exception as e:
        st.error(f"Failed to load {name}: {e}")
        return None


def render_thermal(temp: np.ndarray, vmin: float, vmax: float, cmap_name: str = "inferno") -> Image.Image:
    """Render a temperature matrix as a PIL Image with a chosen colormap and fixed scale."""
    temp_clipped = np.clip(temp, vmin, vmax)
    normed = (temp_clipped - vmin) / (vmax - vmin + 1e-9)
    colored = cm.get_cmap(cmap_name)(normed)  # RGBA 0..1
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def to_bytes(image: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def compute_global_minmax(frames: List[ThermalFrame]) -> Tuple[float, float]:
    mins = [np.nanmin(f.temp) for f in frames]
    maxs = [np.nanmax(f.temp) for f in frames]
    return float(np.min(mins)), float(np.max(maxs))


def make_video(
    frames: List[ThermalFrame],
    vmin: float,
    vmax: float,
    fps: int = 5,
    cmap: str = "inferno",
    as_gif: bool = False,
) -> bytes:
    imgs = [np.array(render_thermal(f.temp, vmin, vmax, cmap)) for f in frames]
    h, w, _ = imgs[0].shape

    if as_gif:
        frames_pil = [Image.fromarray(im) for im in imgs]
        buf = io.BytesIO()
        frames_pil[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames_pil[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        return buf.getvalue()

    # MP4 with OpenCV (requires ffmpeg backend)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = "_tmp_dlt_thermal.mp4"
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for im in imgs:
        out.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    out.release()
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data


# ------------------------------------------------------------------------------------
# PDF builder (updated for Streamlit Cloud)
# ------------------------------------------------------------------------------------
def build_pdf_report(
    frames: List[ThermalFrame],
    vmin: float,
    vmax: float,
    cmap: str,
    template: dict,
) -> bytes:
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    # try unicode first, but it may not have italic on Streamlit Cloud
    font_family = _load_unicode_font_safe(pdf) or "Helvetica"

    # ----- Cover page -----
    pdf.add_page()
    pdf.set_font(font_family, "B", 20)
    pdf.cell(0, 12, template.get("title", "Thermal Analysis Report"), ln=1, align="C")

    pdf.set_font(font_family, "", 12)
    pdf.cell(0, 8, template.get("subtitle", "Generated with DataLens.Tools"), ln=1, align="C")

    # ----- Body pages (2 frames per page) -----
    for i, fr in enumerate(frames):
        # new page for every 2 frames
        if i % 2 == 0:
            pdf.add_page()

        img = render_thermal(fr.temp, vmin, vmax, cmap)
        img_bytes = to_bytes(img, fmt="PNG")

        # positions
        x = 15
        y = 20 if (i % 2 == 0) else 150

        # explicit PNG so fpdf2 doesn’t guess
        img_stream = io.BytesIO(img_bytes)
        pdf.image(img_stream, x=x, y=y, w=90, type="PNG")

        stats = fr.stats

        pdf.set_xy(x + 100, y)
        pdf.set_font(font_family, "B", 12)
        pdf.cell(0, 8, fr.name, ln=1)

        pdf.set_font(font_family, "", 11)
        pdf.set_x(x + 100)
        pdf.multi_cell(
            0,
            6,
            f"Min: {stats['min']:.2f} °C\n"
            f"Max: {stats['max']:.2f} °C\n"
            f"Mean: {stats['mean']:.2f} °C\n"
            f"95th pct: {stats['p95']:.2f} °C",
        )

    # ----- Footer page -----
    pdf.add_page()
    footer_text = template.get(
        "footer",
        "© 2025 DataLens.Tools — Thermal Studio",
    )

    # 🔐 this is the important part:
    # try italic → if the font doesn't have it, fall back to normal
    try:
        pdf.set_font(font_family, "I", 10)
    except Exception:
        pdf.set_font(font_family, "", 10)

    pdf.multi_cell(0, 6, footer_text)

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")


# ------------------------------------------------------------------------------------
# Sidebar (navigation + global scale)
# ------------------------------------------------------------------------------------
with st.sidebar:
    st.title("DataLens Thermal Studio")
    page = st.radio(
        "Navigate",
        ["Home", "Import & Preview", "Batch Processing", "Video Creator", "Report", "Settings"],
    )
    st.caption("MVP • Streamlit Edition")

with st.sidebar:
    st.markdown("---")
    st.subheader("Global Scale")
    vmin_vmax_mode = st.selectbox("Scale mode", ["Auto from batch", "Manual"], index=0)
    if vmin_vmax_mode == "Manual":
        vmin = st.number_input("Min °C", value=20.0)
        vmax = st.number_input("Max °C", value=60.0)
    else:
        vmin = vmax = None  # computed dynamically

    cmap = st.selectbox("Colormap", ["inferno", "plasma", "magma", "turbo", "viridis", "jet"])

    st.markdown(
        '<div class="sidebar-footer">© 2025 '
        '<a href="https://datalens.tools" target="_blank">DataLens.Tools</a>'
        "</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------------------------
if page == "Home":
    st.header("Welcome")
    st.write(
        "This MVP lets you import thermal frames (CSV/NPY or images), "
        "normalize scales, render frames, create videos, and export a PDF report."
    )
    st.info(
        "Tip: For true radiometric FLIR data, export temperature matrices to CSV/NPY "
        "and load them here. If you load plain images, the app will estimate temperatures from grayscale intensities."
    )

elif page == "Import & Preview":
    st.header("Import & Preview")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Upload files")
        min_c = st.number_input("Assumed min °C (for image → temp mapping)", value=20.0)
        max_c = st.number_input("Assumed max °C (for image → temp mapping)", value=60.0)
        files = st.file_uploader(
            "Drop thermal images (PNG/JPG/TIFF) or matrices (CSV/NPY)",
            accept_multiple_files=True,
        )
        if files:
            new_frames = []
            for f in files:
                fr = load_file_as_frame(f, min_c, max_c)
                if fr is not None:
                    new_frames.append(fr)
            st.session_state[SK_FRAMES].extend(new_frames)
            st.success(f"Imported {len(new_frames)} file(s)")

    with c2:
        st.subheader("Batch")
        frames: List[ThermalFrame] = st.session_state[SK_FRAMES]
        st.write(f"Total frames: **{len(frames)}**")
        if frames:
            if vmin is None or vmax is None:
                auto_min, auto_max = compute_global_minmax(frames)
            else:
                auto_min, auto_max = vmin, vmax
            st.session_state[SK_GLOBAL_MINMAX] = (auto_min, auto_max)
            st.write(f"Scale: {auto_min:.2f}–{auto_max:.2f} °C")
            if st.button("Clear batch"):
                st.session_state[SK_FRAMES] = []
                st.experimental_rerun()

    st.markdown("---")
    frames: List[ThermalFrame] = st.session_state[SK_FRAMES]
    if frames:
        vmin_now, vmax_now = st.session_state[SK_GLOBAL_MINMAX]
        grid_cols = st.slider("Preview columns", 2, 5, 3)
        cols = st.columns(grid_cols)
        for i, fr in enumerate(frames):
            with cols[i % grid_cols]:
                img = render_thermal(fr.temp, vmin_now, vmax_now, cmap)
                st.image(
                    img,
                    caption=f"{fr.name} | min {fr.stats['min']:.1f}°C, max {fr.stats['max']:.1f}°C",
                )

elif page == "Batch Processing":
    st.header("Batch Processing")
    frames: List[ThermalFrame] = st.session_state[SK_FRAMES]
    if not frames:
        st.warning("No frames loaded yet. Go to 'Import & Preview' first.")
    else:
        vmin_now, vmax_now = st.session_state[SK_GLOBAL_MINMAX]
        st.write(
            f"Using unified scale: **{vmin_now:.2f}–{vmax_now:.2f} °C** | Colormap: **{cmap}**"
        )

        # export normalized PNGs
        if st.button("Export normalized PNGs (zip)"):
            import zipfile

            tmp = io.BytesIO()
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                for fr in frames:
                    img = render_thermal(fr.temp, vmin_now, vmax_now, cmap)
                    z.writestr(
                        os.path.splitext(fr.name)[0] + "_normalized.png",
                        to_bytes(img),
                    )
            st.download_button(
                "Download PNGs.zip",
                data=tmp.getvalue(),
                file_name="normalized_frames.zip",
            )

        st.markdown("### Table of statistics")
        df = pd.DataFrame([{"file": f.name, **f.stats} for f in frames])
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download stats CSV", data=csv, file_name="thermal_stats.csv")

elif page == "Video Creator":
    st.header("Create Video/GIF from Frames")
    frames: List[ThermalFrame] = st.session_state[SK_FRAMES]
    if not frames:
        st.warning("No frames loaded yet. Go to 'Import & Preview' first.")
    else:
        vmin_now, vmax_now = st.session_state[SK_GLOBAL_MINMAX]
        fps = st.slider("FPS", 1, 30, 6)
        as_gif = st.checkbox("Export as GIF (fallback if MP4 unavailable)")
        if st.button("Render"):
            data = make_video(frames, vmin_now, vmax_now, fps=fps, cmap=cmap, as_gif=as_gif)
            fname = "thermal_animation.gif" if as_gif else "thermal_animation.mp4"
            st.download_button("Download video", data=data, file_name=fname)

elif page == "Report":
    st.header("Generate Report (PDF)")
    frames: List[ThermalFrame] = st.session_state[SK_FRAMES]
    if not frames:
        st.warning("No frames loaded yet. Go to 'Import & Preview' first.")
    else:
        vmin_now, vmax_now = st.session_state[SK_GLOBAL_MINMAX]
        with st.expander("Template settings"):
            t = st.session_state[SK_TEMPLATE]
            t["title"] = st.text_input("Title", t["title"])
            t["subtitle"] = st.text_input("Subtitle", t["subtitle"])
            t["footer"] = st.text_area("Footer", t["footer"])
        if st.button("Build PDF"):
            pdf_bytes = build_pdf_report(
                frames,
                vmin_now,
                vmax_now,
                cmap,
                st.session_state[SK_TEMPLATE],
            )
            st.download_button(
                "Download report.pdf",
                data=pdf_bytes,
                file_name="thermal_report.pdf",
            )

elif page == "Settings":
    st.header("Settings")
    st.write("Configure defaults and export options.")
    st.code(
        """requirements.txt
streamlit
numpy
pillow
matplotlib
opencv-python
pandas
fpdf2
        """,
        language="bash",
    )
    st.caption(
        "Note: MP4 export requires an FFmpeg-enabled OpenCV build on your system. Use GIF if MP4 fails."
    )
