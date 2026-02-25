import os
import cv2
import numpy as np
from PIL import Image

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_svg_path(contours, width, height):
    svg_header = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    path_data = ""
    for cnt in contours:
        if cv2.contourArea(cnt) < 10: continue
        
        # --- RAHASIA MULUS DI SINI ---
        # Kita naikin epsilon-nya dikit (0.001) biar dia gak gampang bikin titik baru
        # Ini bakal nge-force garis jadi kurva panjang, bukan patah-patah
        epsilon = 0.0012 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 3: continue
        
        d = f"M {approx[0][0][0]} {approx[0][0][1]} "
        for pt in approx[1:]:
            d += f"L {pt[0][0]} {pt[0][1]} "
        d += "Z "
        path_data += f'<path d="{d}" fill="black" stroke="black" stroke-width="0.5" stroke-linejoin="round" />\n'
    
    return svg_header + path_data + "</svg>"

def process_image():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")): continue

        name = os.path.splitext(file)[0]
        input_path = os.path.join(INPUT_DIR, file)
        print(f"🚀 Smoothing: {file}")

        # 1. Load & Heavy Blur (Biar pixel "tangga" ilang)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # Gedein resolusi biar tracing lebih luas ruang geraknya
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        
        # Blur tipis buat ngeratain pinggiran yang tajem
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 2. Thresholding
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Tracing dengan mode KCOS (Terbaik buat ornamen)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPRO_TC89_KCOS)
        
        # 4. Save SVG
        h, w = bw.shape
        svg_content = create_svg_path(contours, w, h)
        
        with open(os.path.join(OUTPUT_DIR, f"{name}.svg"), "w") as f:
            f.write(svg_content)
            
        print(f"✅ Selesai! Coba cek di Corel, harusnya udah gak gatel lagi liatnya.")

if __name__ == "__main__":
    process_image()
