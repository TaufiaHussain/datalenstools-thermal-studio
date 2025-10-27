#  DataLens Thermal Studio — Streamlit MVP

A **Streamlit-based thermal imaging studio** that replicates key workflows from **FLIR Thermal Studio**, built for rapid batch analysis, visualization, and reporting of thermal data.

Developed by **[DataLens.Tools](https://datalens.tools)** — empowering researchers and engineers with open, AI-powered data analysis tools.



##  Overview

This MVP provides a **desktop-like web interface** for analyzing thermal data (images or temperature matrices).  
It supports unified color normalization, quick statistics, video generation, and PDF reporting — all within Streamlit.

###  Core Features

-  **Batch Import**  
  Upload multiple **thermal images (PNG/JPG/TIFF)** or **temperature matrices (CSV/NPY)** at once.
  
-  **Global Normalization**  
  Auto or manual scale normalization (°C range) across all frames.
  
-  **Quick Analytics**  
  Calculate min, max, mean, and 95th percentile temperatures per frame.
  
-  **Video Creator**  
  Export animated MP4 or GIF sequences from batches.
  
-  **PDF Report Generator**  
  Build clean, professional thermal reports with title, subtitle, and footer templates.

-  **Unified Colormap Control**  
  Supports Inferno, Plasma, Magma, Turbo, Viridis, Jet, and more.



##  Screenshot

*(optional: add a screenshot later)*  
```
![App Screenshot](docs/screenshot.png)
```



##  Installation

### 1️ Clone the Repository

```bash
git clone https://github.com/DataLensTools/datalens-thermal-studio.git
cd datalens-thermal-studio
```

### 2️ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # on macOS/Linux
venv\Scripts\activate     # on Windows
```

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️ Run the App

```bash
streamlit run app.py
```



##  Requirements

```
streamlit
numpy
pillow
matplotlib
opencv-python
pandas
fpdf2
```

> **Note:** MP4 export requires an FFmpeg-enabled OpenCV build on your system.  
> If unavailable, you can export GIFs instead.



##  File Structure

```
📦 datalens-thermal-studio/
 ┣ 📜 app.py                ← Main Streamlit app
 ┣ 📜 requirements.txt      ← Python dependencies
 ┣ 📁 sample_data/          ← Example thermal images or CSVs (optional)
 ┣ 📁 docs/                 ← Screenshots or documentation
 ┗ 📜 README.md             ← You’re here
```



##  Example Workflow

1. **Import & Preview**  
   Upload your thermal images (PNG/JPG/TIFF) or numerical matrices (CSV/NPY).  
   The app converts images to temperature estimates using a linear grayscale → °C mapping.

2. **Batch Processing**  
   Compute per-frame statistics and export normalized PNGs or CSV summaries.

3. **Video Creator**  
   Combine frames into an MP4 or GIF for temporal visualization.

4. **PDF Report**  
   Generate and download a full analysis report (`thermal_report.pdf`).



##  Architecture

- **Frontend/UI:** Streamlit  
- **Rendering:** Matplotlib Colormaps  
- **Image I/O:** Pillow (PIL)  
- **Video Encoding:** OpenCV  
- **Reporting:** FPDF2  
- **Core Logic:** Python Dataclasses for frame management



##  Example Code Snippet

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ThermalFrame:
    name: str
    temp: np.ndarray  # temperature matrix in °C

    @property
    def stats(self):
        t = self.temp
        return {
            "min": float(np.nanmin(t)),
            "max": float(np.nanmax(t)),
            "mean": float(np.nanmean(t)),
            "p95": float(np.nanpercentile(t, 95)),
        }
```



##  Author

**Taufia Hussain** (Maintainer) — [LinkedIn](https://www.linkedin.com/in/taufia-hussain-phd-52300015/) · [GitHub](https://github.com/TaufiaHussain) · info@datalens.tools · datalenstools@gmail.com

**Developed by:** [DataLens.Tools](https://datalens.tools)  
**© 2025 DataLens.Tools** — Open source for research and educational use.



##  License

This project is licensed under the **MIT License**.  
You’re free to use, modify, and distribute it with attribution.

```
MIT License
Copyright (c) 2025 DataLens.Tools
```



## 🌐 Links

- 🌍 Website: [https://datalens.tools](https://datalens.tools)  
- 🧑‍💻 GitHub: [DataLensTools](https://github.com/TaufiaHussain/datalenstools-thermal-studio)  
- ✉️ Contact: info@datalens.tools  



###  “Analyze, visualize, and report — all in one thermal studio.”
