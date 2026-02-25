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

        # 1. LOAD & UPSCALING (Set ke 333 DPI / Perbesar Resolusi)
        img = Image.open(input_path)
        w, h = img.size
        # Simulasi naik resolusi (Upscale 2x - 3x biar garis makin tajem)
        new_size = (w * 3, h * 3)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # 2. INVERT LOGIC (Kalau warna dasarnya hitam, kita balik)
        # Kita cek rata-rata pixel, kalau gelap, kita invert biar line-art nya jadi item
        img_temp = img.convert("L")
        stat = np.array(img_temp).mean()
        if stat < 127: # Artinya dominan gelap/hitam
            img = ImageOps.invert(img.convert("RGB"))
            print("🔄 Inverting Colors (Black Background Detected)")

        # 3. CONVERT TO GRAYSCALE
        img = img.convert("L")

        # 4. MEDIUM CONTRAST (Biar garis makin tegas)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5) # Naikin kontras 50%

        # 5. CONVERT TO BLACK & WHITE (1-Bit Line Art Mode)
        arr = np.array(img)
        _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. TRACE BITMAP (High Quality Outline Trace)
        # Bersihin noise dikit sebelum di-trace
        kernel = np.ones((3,3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        h_bw, w_bw = bw.shape
        drawing = Drawing(w_bw, h_bw)

        path_count = 0
        for cnt in contours:
            # Filter objek kecil banget (noise)
            if cv2.contourArea(cnt) < 150:
                continue

            # Smoothing (Biar kayak paha girlband tadi, bray!)
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            p = Path()
            x0, y0 = approx[0][0]
            p.moveTo(x0, h_bw - y0)

            for pt in approx[1:]:
                x, y = pt[0]
                p.lineTo(x, h_bw - y)

            p.closePath()
            p.strokeColor = colors.black
            p.fillColor = colors.black # Outline Trace Style
            p.strokeWidth = 0.5
            drawing.add(p)
            path_count += 1

        # 7. EXPORT EPS & SVG
        eps_out = os.path.join(OUTPUT_DIR, f"{name}.eps")
        svg_out = os.path.join(OUTPUT_DIR, f"{name}.svg")
        renderPS.drawToFile(drawing, eps_out)
        renderSVG.drawToFile(drawing, svg_out)

        print(f"✅ Berhasil! {path_count} Path dibuat.\n")

if __name__ == "__main__":
    process_image()
    print("=== SEMUA BERES, BOSS! ===")
