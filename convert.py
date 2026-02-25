import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_svg_path(contours, width, height):
    # Bikin template SVG manual biar nggak dirusak ReportLab
    svg_header = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    path_data = ""
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 15: continue
        
        # Smoothing buat ornamen biar nggak kaku
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
        print(f"🚀 Processing: {file}")

        # 1. Load & Boost Contrast
        img = Image.open(input_path).convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(3.0)
        arr = np.array(img)

        # 2. Logic Invert (Penting buat Ornamen)
        # Kalo background item, kita balik biar objeknya item
        if arr.mean() < 127:
            arr = cv2.bitwise_not(arr)

        # 3. Thresholding (Bikin Line Art B&W)
        _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Tracing Contours
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        # 5. Save as SVG (Manual Mode - Anti Meteor)
        h, w = bw.shape
        svg_content = create_svg_path(contours, w, h)
        
        svg_out = os.path.join(OUTPUT_DIR, f"{name}.svg")
        with open(svg_out, "w") as f:
            f.write(svg_content)
            
        print(f"✅ Mantap! File SVG udah kelar. Cek sizenya, pasti lebih gede dari 1KB!")

if __name__ == "__main__":
    process_image()
