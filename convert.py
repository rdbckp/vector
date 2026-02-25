import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_svg_path(contours, width, height):
    svg_header = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    path_data = ""
    for cnt in contours:
        if cv2.contourArea(cnt) < 20: continue
        
        # SMOOTHING LEVEL: Makin kecil epsilon, makin detail. 
        # 0.0005 itu sweet spot buat ornamen lekuk.
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
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
        print(f"🚀 Glow-up: {file}")

        # 1. UPSCALING (Gedein 3x biar pixel-nya rapet)
        img = Image.open(input_path).convert("L")
        w, h = img.size
        img = img.resize((w * 3, h * 3), Image.Resampling.LANCZOS)
        
        # 2. ANTI-GRADAKAN FILTER (Gaussian Blur)
        # Rahasianya di sini: dibikin agak ngeblur dikit biar "geriginya" ilang
        arr = np.array(img)
        arr = cv2.GaussianBlur(arr, (5, 5), 0)

        # 3. KONTRAS JAHAT
        _, bw = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. MORPHOLOGY (Biar garisnya nyambung & solid)
        kernel = np.ones((3,3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        # 5. TRACING (Pake mode KCOS buat kurva)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # 6. SAVE SVG
        h_bw, w_bw = bw.shape
        svg_content = create_svg_path(contours, w_bw, h_bw)
        
        with open(os.path.join(OUTPUT_DIR, f"{name}.svg"), "w") as f:
            f.write(svg_content)
            
        print(f"✅ Ornamen siap! Udah mulus kayak paha... personil girlband. 😎")

if __name__ == "__main__":
    process_image()
