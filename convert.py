import os
import numpy as np
import cv2
from PIL import Image
from skimage import filters, morphology, measure
from reportlab.graphics import renderPS, renderSVG
from reportlab.graphics.shapes import Drawing, Path
from reportlab.lib import colors

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
    arr = cv2.GaussianBlur(arr, (5,5), 0)
    arr = cv2.equalizeHist(arr)

    # ===== AUTO THRESHOLD =====
    thresh = filters.threshold_otsu(arr)
    bw = (arr > thresh).astype(np.uint8)

    # ===== CLEANUP =====
    bw = morphology.remove_small_objects(bw.astype(bool), min_size=300)
    bw = morphology.binary_closing(bw, morphology.disk(2))
    bw = bw.astype(np.uint8)

    # ===== EDGE =====
    edges = cv2.Canny((bw*255).astype(np.uint8), 60, 140)

    # ===== TRACE =====
    contours = measure.find_contours(edges, 0.5)

    h, w = edges.shape
    drawing = Drawing(w, h)

    path_count = 0

    for contour in contours:
        if len(contour) < 40:
            continue

        p = Path()
        p.moveTo(contour[0][1], h - contour[0][0])

        for y, x in contour[1:]:
            p.lineTo(x, h - y)

        p.strokeColor = colors.black
        p.fillColor = None
        p.strokeWidth = 0.6

        drawing.add(p)
        path_count += 1

    # ===== EXPORT =====
    eps_out = os.path.join(OUTPUT_DIR, f"{name}.eps")
    svg_out = os.path.join(OUTPUT_DIR, f"{name}.svg")

    renderPS.drawToFile(drawing, eps_out)
    renderSVG.drawToFile(drawing, svg_out)

    print("Generated:", eps_out)
    print("Generated:", svg_out)
    print("Paths:", path_count)

print("\n=== ALL FILES PROCESSED ===")
