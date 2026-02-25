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

        # 1. LOAD & UPSCALING (Gedein biar detail dapet)
        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        img = img.resize((w * 3, h * 3), Image.Resampling.LANCZOS)

        # 2. CORE LOGIC (Grayscale & Contrast)
        gray = ImageOps.grayscale(img)
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.0) # Makin galak kontrasnya
        
        arr = np.array(gray)

        # 3. AUTO-INVERT SAKTI
        # Cek pojok kiri atas (biasanya background). Kalau terang, berarti objeknya gelap.
        bg_sample = arr[0:10, 0:10].mean()
        if bg_sample > 127: 
            # Background putih -> Objek hitam (Normal)
            _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Background hitam -> Objek terang (Invert)
            _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. TRACING (Outline Mode)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_bw, w_bw = bw.shape
        drawing = Drawing(w_bw, h_bw)

        path_count = 0
        for cnt in contours:
            # Saring yang beneran kecil doang (di bawah 10 pixel area)
            if cv2.contourArea(cnt) < 10: 
                continue

            # Mulusin dikit biar nggak kaku
            epsilon = 0.0004 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            p = Path()
            x0, y0 = approx[0][0]
            p.moveTo(x0, h_bw - y0)

            for pt in approx[1:]:
                x, y = pt[0]
                p.lineTo(x, h - y if 'h' in locals() else h_bw - y) # Fix koordinat

            p.closePath()
            p.fillColor = colors.black # Biar muncul bentuk solid di Corel
            p.strokeColor = None 
            drawing.add(p)
            path_count += 1

        if path_count == 0:
            print(f"⚠️ Waduh! Gak ada objek ketemu di {file}. Cek gambarnya bray!")
            continue

        # 5. EXPORT
        eps_out = os.path.join(OUTPUT_DIR, f"{name}.eps")
        svg_out = os.path.join(OUTPUT_DIR, f"{name}.svg")
        renderPS.drawToFile(drawing, eps_out)
        renderSVG.drawToFile(drawing, svg_out)
        print(f"✅ Mantap! {path_count} objek berhasil di-trace.")

if __name__ == "__main__":
    process_image()
