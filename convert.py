import os
import cv2
import numpy as np

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_svg_path(contours, width, height):
    svg_header = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    path_data = ""
    
    for cnt in contours:
        # Filter debu kecil banget
        if cv2.contourArea(cnt) < 10: continue
        
        # EPILON DITINGGIIN BIAR MULUS LICIN (0.002)
        # Makin gede angka ini, makin dikit titiknya, makin "sliding" garisnya
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 3: continue
        
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
        print(f"🚀 Scanning: {file}")

        # 1. LOAD MURNI & GEDEIN
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Upscale 3x biar tracingnya halus
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)

        # 2. SIMPLE THRESHOLD (Gak pake ribet)
        # Kita paksa: yang agak gelap jadi item, yang terang jadi putih.
        # Kalau gambar loe background putih objek item, pake THRESH_BINARY_INV
        _, bw = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

        # 3. CEK KALO KOSONG (Kalo kosong, coba mode normal tanpa INV)
        if cv2.countNonZero(bw) == 0:
            _, bw = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # 4. TRACING (Sikat Miring!)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. SAVE SVG
        h, w = bw.shape
        svg_content = create_svg_path(contours, w, h)
        
        svg_path = os.path.join(OUTPUT_DIR, f"{name}.svg")
        with open(svg_path, "w") as f:
            f.write(svg_content)
            
        print(f"✅ Status: {len(contours)} objek ditemukan | Size: {os.path.getsize(svg_path)} Bytes")

if __name__ == "__main__":
    process_image()
