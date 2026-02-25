import os
import cv2
import numpy as np

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_svg_path(contours, width, height):
    # Header SVG standar
    svg_header = f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    path_data = ""
    
    for cnt in contours:
        # Filter objek super kecil (debu)
        if cv2.contourArea(cnt) < 15: 
            continue
        
        # --- RAHASIA MULUS JILID 2 ---
        # Kita pake epsilon yang agak berani (0.0015) 
        # Biar garis yang "gerigi" dipaksa jadi lurus/lengkung panjang
        epsilon = 0.0015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 3: 
            continue
        
        # Rakit koordinat
        d = f"M {approx[0][0][0]} {approx[0][0][1]} "
        for pt in approx[1:]:
            d += f"L {pt[0][0]} {pt[0][1]} "
        d += "Z "
        
        # Tambahin style biar makin smooth pas di-render
        path_data += f'<path d="{d}" fill="black" stroke="black" stroke-width="0.2" stroke-linejoin="round" />\n'
    
    return svg_header + path_data + "</svg>"

def process_image():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")): 
            continue

        name = os.path.splitext(file)[0]
        input_path = os.path.join(INPUT_DIR, file)
        print(f"🚀 Tracing: {file}")

        # 1. Load Grayscale
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Gagal baca file: {file}")
            continue
            
        # 2. Upscale + Blur (Biar gak tajam berlebihan)
        # Kita gedein 3x biar si tracing punya banyak data buat di-smooth
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # 3. Thresholding (Bikin B&W)
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Tracing (Pake CHAIN_APPROX_SIMPLE - Paling Aman sedunia)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Save ke SVG
        h, w = bw.shape
        svg_content = create_svg_path(contours, w, h)
        
        svg_path = os.path.join(OUTPUT_DIR, f"{name}.svg")
        with open(svg_path, "w") as f:
            f.write(svg_content)
            
        print(f"✅ Mantap Boss! SVG siap di-cek. Size: {os.path.getsize(svg_path)/1024:.2f} KB")

if __name__ == "__main__":
    process_image()
