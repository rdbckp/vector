import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from reportlab.graphics import renderPS, renderSVG
from reportlab.graphics.shapes import Drawing, Path
from reportlab.lib import colors

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_image():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        name = os.path.splitext(file)[0]
        input_path = os.path.join(INPUT_DIR, file)
        print(f"🚀 Processing: {file}")

        # 1. LOAD & STRETCH (Biar detail ornamen nggak ilang)
        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        # Gedein 2x aja biar nggak kegedean filenya
        img = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        
        # 2. PRE-PROCESS (Corel Logic)
        gray = ImageOps.grayscale(img)
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.5) # Kontras galak
        
        arr = np.array(gray)

        # 3. THRESHOLD (Black & White tegas)
        # Kita pake Simple Binary aja biar nggak leaking
        _, bw = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY_INV)

        # 4. TRACING (High Quality Mode)
        # Pake CHAIN_APPROX_TC89_KCOS buat dapet lekukan bezier-like
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        h_bw, w_bw = bw.shape
        drawing = Drawing(w_bw, h_bw)

        path_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < 20: 
                continue

            # Smoothing level: Low epsilon = High Detail
            epsilon = 0.0002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            p = Path(fillColor=colors.black, strokeColor=None)
            
            # Start Point
            x0, y0 = approx[0][0]
            p.moveTo(x0, h_bw - y0) # Flip Y untuk ReportLab

            # Draw Lines
            for pt in approx[1:]:
                x, y = pt[0]
                p.lineTo(x, h_bw - y)

            p.closePath()
            drawing.add(p)
            path_count += 1

        if path_count == 0:
            print(f"⚠️ Objek gak deteksi di {file}")
            continue

        # 5. EXPORT (EPS & SVG)
        eps_out = os.path.join(OUTPUT_DIR, f"{name}.eps")
        svg_out = os.path.join(OUTPUT_DIR, f"{name}.svg")
        
        # Render manual biar gak corrupt di Corel X5
        renderPS.drawToFile(drawing, eps_out)
        renderSVG.drawToFile(drawing, svg_out)
        
        print(f"✅ Mantap! {path_count} ornamen jadi vektor mulus.")

if __name__ == "__main__":
    process_image()
