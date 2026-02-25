import cv2
import numpy as np
from PIL import Image
from skimage import filters, morphology, measure
from reportlab.graphics import renderPS
from reportlab.graphics.shapes import Drawing, Path
from reportlab.lib import colors
import os

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    name = os.path.splitext(file)[0]
    input_path = os.path.join(INPUT_DIR, file)

    # ===== LOAD =====
    img = Image.open(input_path).convert("L")
    arr = np.array(img)

    # ===== PREPROCESS =====
    # blur
    arr = cv2.GaussianBlur(arr, (5,5), 0)

    # contrast boost
    arr = cv2.equalizeHist(arr)

    # adaptive threshold
    thresh = filters.threshold_otsu(arr)
    bw = (arr > thresh).astype(np.uint8)

    # morphology cleanup
    bw = morphology.remove_small_objects(bw.astype(bool), min_size=200)
    bw = morphology.binary_closing(bw, morphology.disk(2))
    bw = bw.astype(np.uint8)

    # ===== EDGE DETECT =====
    edges = cv2.Canny((bw*255).astype(np.uint8), 80, 160)

    # ===== CONTOUR TRACE =====
    contours = measure.find_contours(edges, 0.5)

    h, w = edges.shape
    drawing = Drawing(w, h)

    total_paths = 0

    for contour in contours:
        if len(contour) < 30:
            continue

        p = Path()
        p.moveTo(contour[0][1], h - contour[0][0])

        for y, x in contour[1:]:
            p.lineTo(x, h - y)

        # continuous path for cutting
        p.strokeColor = colors.black
        p.fillColor = None
        p.strokeWidth = 0.5

        drawing.add(p)
        total_paths += 1

    # ===== EXPORT =====
    eps_out = os.path.join(OUTPUT_DIR, f"{name}.eps")
    svg_out = os.path.join(OUTPUT_DIR, f"{name}.svg")

    # EPS
    renderPS.drawToFile(drawing, eps_out)

    # SVG
    from reportlab.graphics import renderSVG
    renderSVG.drawToFile(drawing, svg_out)

    print("Generated:", eps_out)
    print("Generated:", svg_out)
    print("Paths:", total_paths)

print("\n=== DONE ALL FILES ===")
