import os
import cv2
import numpy as np
from PIL import Image, ImageOps

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_svg_path(contours, width, height):
    svg_header = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    path_data = ""
    for cnt in contours:
        # Jangan terlalu pelit sama area, biar garis tipis nggak ilang
        if cv2.contourArea(cnt) < 5: continue
        
        # Smoothing yang beneran "paha girlband" tapi nggak ngilangin bentuk
        epsilon = 0.0003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        d = f"M {approx[0][0][0]} {approx[0][0][1]} "
        for pt in approx[1:]:
            d += f"L {pt[0][0]} {pt[0][1]} "
        d += "Z "
        path_data += f'<path d="{d}" fill="black" />\n'
    return svg_header + path_data + "</svg>"

def process_image():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")): continue

        name = os.path.splitext(file)[0]
        input_path = os.path.join(INPUT_DIR, file)
        print(f"🚀 Tracing Ulang: {file}")

        # 1. LOAD MURNI
        img_pil = Image.open(input_path).convert("L")
        w, h = img_pil.size
        
        # Upscale 2x aja biar gak berat tapi tetep tajem
        img_pil = img_pil.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        arr = np.array(img_pil)

        # 2. ANTI-GAIB: Adaptive Threshold
        # Ini bakal nangkep garis tipis di area manapun tanpa pandang bulu
        bw = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

        # 3. CLEANING (Buat ngilangin bintik kecil tapi garis utama aman)
        kernel = np.ones((2,2), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        # 4. TRACING KELAS BERAT
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        # 5. SAVE SVG
        h_bw, w_bw = bw.shape
        svg_content = create_svg_path(contours, w_bw, h_bw)
        
        svg_file = os.path.join(OUTPUT_DIR, f"{name}.svg")
        with open(svg_file, "w") as f:
            f.write(svg_content)
            
        # Cek size buat mastiin
        file_size = os.path.getsize(svg_file) / 1024
        print(f"✅ Kelar! Size: {file_size:.2f} KB | Path: {len(contours)}")

if __name__ == "__main__":
    process_image()
