from PIL import Image, ImageOps
import numpy as np
from skimage import measure
from reportlab.graphics import renderPS
from reportlab.graphics.shapes import Drawing, Path

INPUT = "input.png"
OUTPUT = "output_vector.eps"

# Load image
img = Image.open(INPUT).convert("L")

# Threshold
bw = ImageOps.invert(img).point(lambda x: 255 if x > 120 else 0, '1')
arr = np.array(bw, dtype=np.uint8)

# Find contours
contours = measure.find_contours(arr, 0.5)

h, w = arr.shape
drawing = Drawing(w, h)

for contour in contours:
    p = Path()
    p.moveTo(contour[0][1], h - contour[0][0])
    for y, x in contour[1:]:
        p.lineTo(x, h - y)
    p.closePath()
    p.strokeColor = None
    p.fillColor = None
    p.strokeWidth = 1
    drawing.add(p)

# Save EPS
renderPS.drawToFile(drawing, OUTPUT)
print("Saved:", OUTPUT)
