# Real-time-Webcam-Image-Processing-Studio
### Python + Flask + OpenCV + NumPy

## Project Description
An interactive real-time image processing application that captures
live webcam feed and applies 25+ image processing techniques using
Python and OpenCV. 

## Tech Stack
- **Python 3.x** — core language
- **OpenCV (cv2)** — all image processing algorithms
- **NumPy** — array/matrix operations
- **Flask** — lightweight web server / backend API
- **HTML/CSS/JS** — frontend UI (runs in browser)

## How to Run

### Step 1 — Install dependencies
Open terminal/command prompt in the project folder and run:
```
pip install -r requirements.txt
```

### Step 2 — Start the server
```
python app.py
```

### Step 3 — Open in browser
Go to: **http://127.0.0.1:5000**

### Step 4 — Use the app
1. Click **"Start Camera"** and allow camera permission
2. Click any filter from the sidebar
3. Watch your live feed processed in real-time by Python/OpenCV
4. Adjust parameters using the sliders
5. Click **"Save Frame"** to download a processed image

## Filters Implemented (25+)

### Image Enhancement
| Filter | OpenCV / NumPy function used |
|---|---|
| Grayscale | cv2.cvtColor |
| Negative | cv2.bitwise_not |
| Contrast Stretch | NumPy where |
| Sepia | cv2.transform |
| Threshold | cv2.threshold |
| Gray Level Slicing | NumPy masking |
| Power Law (Gamma) | cv2.LUT |
| Histogram Equalization | cv2.equalizeHist |
| Sharpening | cv2.filter2D |
| Smoothing | cv2.GaussianBlur |
| Median Filter | cv2.medianBlur |
| Emboss | cv2.filter2D |
| Pencil Sketch | cv2.divide |

### Edge Detection
| Filter | OpenCV / NumPy function used |
|---|---|
| Sobel | cv2.Sobel |
| Robert Cross | cv2.filter2D (custom kernel) |
| Prewitt | cv2.filter2D (custom kernel) |
| Laplacian | cv2.Laplacian |
| Canny | cv2.Canny |
| LoG | cv2.GaussianBlur + cv2.Laplacian |

### Morphological Operations
| Filter | OpenCV function used |
|---|---|
| Dilation | cv2.dilate |
| Erosion | cv2.erode |
| Opening | cv2.morphologyEx OPEN |
| Closing | cv2.morphologyEx CLOSE |
| Boundary Extraction | erode + subtract |
| Skeletonization | iterative erode + open |

###  Segmentation
| Filter | Method |
|---|---|
| Otsu Threshold | cv2.THRESH_OTSU |
| Binary Segmentation | cv2.threshold |
| Color Quantization | NumPy floor division |
| Region Growing | BFS pixel traversal |

## Project Architecture
```
image_processing_studio/
├── app.py              ← Python Flask server + all OpenCV filters
├── requirements.txt    ← Dependencies
├── README.md           ← This file
└── templates/
    └── index.html      ← Frontend UI (HTML/CSS/JS)
```

**Flow:**
Browser (webcam) → captures frame → sends to Python Flask server
→ OpenCV processes it → sends result back → Browser displays it

---

