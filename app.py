from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import re

app = Flask(__name__)


# ─── Image Enhancement ───────────────────────────────────────────────

def apply_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_negative(img):
    return cv2.bitwise_not(img)

def apply_contrast_stretch(img, lo=30, hi=220):
    result = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        stretched = np.where(ch <= lo, 0,
                    np.where(ch >= hi, 255,
                    (ch - lo) / (hi - lo) * 255))
        result[:, :, c] = np.clip(stretched, 0, 255)
    return result.astype(np.uint8)

def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img.astype(np.float32), kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_threshold(img, thresh=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def apply_gray_level_slicing(img, lo=80, hi=180):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sliced = np.where((gray >= lo) & (gray <= hi), 255, gray * 0.3).astype(np.uint8)
    return cv2.cvtColor(sliced, cv2.COLOR_GRAY2BGR)

def apply_power_law(img, gamma=1.0):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, lut)

def apply_hist_equalize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_sharpen(img, strength=1.0):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    base   = cv2.filter2D(img, -1, kernel)
    return cv2.addWeighted(img, 1 - strength * 0.5, base, strength * 0.5, 0)

def apply_smooth(img, ksize=5):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_median(img, ksize=5):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, ksize)

def apply_emboss(img):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]], dtype=np.float32)
    emboss = cv2.filter2D(gray, -1, kernel) + 128
    emboss = np.clip(emboss, 0, 255).astype(np.uint8)
    return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)

def apply_pencil_sketch(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (21, 21), 0)
    sketch  = cv2.divide(gray, blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# ─── Edge Detection ───────────────────────────────────────────────────

def apply_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx**2 + gy**2)
    mag  = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

def apply_robert(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kx    = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky    = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx    = cv2.filter2D(gray, -1, kx)
    gy    = cv2.filter2D(gray, -1, ky)
    mag   = np.sqrt(gx**2 + gy**2)
    mag   = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

def apply_prewitt(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kx    = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky    = np.array([[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]], dtype=np.float32)
    gx    = cv2.filter2D(gray, -1, kx)
    gy    = cv2.filter2D(gray, -1, ky)
    mag   = np.sqrt(gx**2 + gy**2)
    mag   = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

def apply_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    lap  = np.abs(lap)
    lap  = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def apply_canny(img, low=50, high=150):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_log(img, sigma=2.0):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred= cv2.GaussianBlur(gray, (0, 0), sigma)
    log    = cv2.Laplacian(blurred, cv2.CV_64F)
    log    = np.abs(log)
    log    = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(log, cv2.COLOR_GRAY2BGR)


# ─── Morphological Operations ────────────────────────────────────────

def get_kernel(ksize):
    ksize = int(ksize) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

def apply_dilation(img, ksize=2):
    return cv2.dilate(img, get_kernel(ksize), iterations=1)

def apply_erosion(img, ksize=2):
    return cv2.erode(img, get_kernel(ksize), iterations=1)

def apply_opening(img, ksize=2):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel(ksize))

def apply_closing(img, ksize=2):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel(ksize))

def apply_boundary(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded  = cv2.erode(gray, kernel)
    boundary= cv2.subtract(gray, eroded)
    return cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)

def apply_skeleton(img):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    skel   = np.zeros_like(bin_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp   = bin_img.copy()
    for _ in range(20):
        eroded  = cv2.erode(temp, kernel)
        opened  = cv2.dilate(eroded, kernel)
        sub     = cv2.subtract(temp, opened)
        skel    = cv2.bitwise_or(skel, sub)
        temp    = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)


# ─── Segmentation ─────────────────────────────────────────────────────

def apply_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def apply_binary_seg(img, thresh=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def apply_color_quant(img, levels=4):
    factor = 256 // levels
    quantized = (img.astype(np.float32) / factor).astype(np.uint8) * factor
    return quantized

def apply_region_growing(img, seed_x_pct=0.5, tolerance=20):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W     = gray.shape
    sx, sy   = int(seed_x_pct * W), H // 2
    sx       = np.clip(sx, 0, W-1)
    seed_val = int(gray[sy, sx])
    visited  = np.zeros((H, W), dtype=bool)
    result   = np.zeros((H, W), dtype=np.uint8)
    queue    = [(sx, sy)]
    visited[sy, sx] = True
    while queue:
        cx, cy = queue.pop(0)
        result[cy, cx] = 255
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx]:
                if abs(int(gray[ny, nx]) - seed_val) <= tolerance:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ─── Filter Dispatcher ────────────────────────────────────────────────────────

FILTER_MAP = {
    # Enhancement
    "Original"        : lambda img, p: img.copy(),
    "Grayscale"       : lambda img, p: apply_grayscale(img),
    "Negative"        : lambda img, p: apply_negative(img),
    "Contrast Stretch": lambda img, p: apply_contrast_stretch(img, p.get('lo',30), p.get('hi',220)),
    "Sepia"           : lambda img, p: apply_sepia(img),
    "Threshold"       : lambda img, p: apply_threshold(img, p.get('thresh',128)),
    "Gray Slicing"    : lambda img, p: apply_gray_level_slicing(img, p.get('lo',80), p.get('hi',180)),
    "Gamma Correct"   : lambda img, p: apply_power_law(img, p.get('gamma',1.0)),
    "Hist Equalize"   : lambda img, p: apply_hist_equalize(img),
    "Sharpen"         : lambda img, p: apply_sharpen(img, p.get('strength',1.0)),
    "Smooth"          : lambda img, p: apply_smooth(img, p.get('ksize',5)),
    "Median Filter"   : lambda img, p: apply_median(img, p.get('ksize',5)),
    "Emboss"          : lambda img, p: apply_emboss(img),
    "Pencil Sketch"   : lambda img, p: apply_pencil_sketch(img),
    # Edge Detection
    "Sobel"           : lambda img, p: apply_sobel(img),
    "Robert"          : lambda img, p: apply_robert(img),
    "Prewitt"         : lambda img, p: apply_prewitt(img),
    "Laplacian"       : lambda img, p: apply_laplacian(img),
    "Canny"           : lambda img, p: apply_canny(img, p.get('low',50), p.get('high',150)),
    "LoG"             : lambda img, p: apply_log(img, p.get('sigma',2.0)),
    # Morphology
    "Dilation"        : lambda img, p: apply_dilation(img, p.get('ksize',2)),
    "Erosion"         : lambda img, p: apply_erosion(img, p.get('ksize',2)),
    "Opening"         : lambda img, p: apply_opening(img, p.get('ksize',2)),
    "Closing"         : lambda img, p: apply_closing(img, p.get('ksize',2)),
    "Boundary"        : lambda img, p: apply_boundary(img),
    "Skeleton"        : lambda img, p: apply_skeleton(img),
    # Segmentation
    "Otsu"            : lambda img, p: apply_otsu(img),
    "Binary Seg"      : lambda img, p: apply_binary_seg(img, p.get('thresh',128)),
    "Color Quant"     : lambda img, p: apply_color_quant(img, p.get('levels',4)),
    "Region Growing"  : lambda img, p: apply_region_growing(img, p.get('seed',0.5), p.get('tol',20)),
}


def decode_image(data_url):
    """Decode base64 image from browser canvas."""
    header, encoded = data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def encode_image(img):
    """Encode OpenCV image to base64 PNG."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

def compute_histogram(img):
    """Compute grayscale histogram data for charting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    hist = (hist / hist.max() * 100).tolist() if hist.max() > 0 else hist.tolist()
    return [round(v, 1) for v in hist]


# ─── Flask Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data       = request.json
    filter_name= data.get('filter', 'Original')
    params     = data.get('params', {})
    frame_b64  = data.get('frame')

    if not frame_b64:
        return jsonify({'error': 'No frame'}), 400

    img = decode_image(frame_b64)
    if img is None:
        return jsonify({'error': 'Decode failed'}), 400

    fn = FILTER_MAP.get(filter_name)
    if fn is None:
        return jsonify({'error': f'Unknown filter: {filter_name}'}), 400

    try:
        result = fn(img, params)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'result'    : encode_image(result),
        'hist_in'   : compute_histogram(img),
        'hist_out'  : compute_histogram(result),
        'width'     : img.shape[1],
        'height'    : img.shape[0],
    })

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Real-time Image Processing Studio")
    print("  Python + Flask + OpenCV + NumPy")
    print("="*55)
    print("  Open your browser and go to:")
    print("  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
