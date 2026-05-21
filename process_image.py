import os
import sys
import cv2
import numpy as np
from PIL import Image

def get_env_variables():
    image_name = os.environ.get("IMAGE_NAME", "image.png")
    process_mode = os.environ.get("PROCESS_MODE", "upscale")
    return image_name, process_mode

def process_upscale(img):
    # Menggunakan interpolasi Kubik untuk upscale cepat 2x lipat ala nano-banana
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

def process_vector_bw(img):
    # Mengubah ke grayscale -> adaptive thresholding untuk efek tracing vector tajam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filter biner untuk menghasilkan hitam putih tegas tanpa gradasi abu-abu
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    return bw

def process_cartoon(img):
    # Menggunakan Bilateral Filter untuk menghaluskan warna tapi menjaga edge tetap tajam
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # Membuat line art / outline pinggiran
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 9, 2)
    # Gabungkan outline hitam dengan warna yang sudah dihaluskan
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def process_3d_style(img):
    # Meniru efek kedalaman 3D/Pop-art dengan sedikit pergeseran kanal warna (Anaglyph/Emboss effect nhẹ)
    # Di sini kita naikkan kontras, saturasi, dan beri sedikit efek pencahayaan (Look up Table sederhana)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    emboss_kernel = np.array([[ -2, -1, 0],
                              [ -1,  1, 1],
                              [  0,  1, 2]])
    emboss = cv2.filter2D(blur, -1, emboss_kernel)
    # Campurkan gambar asli dengan efek timbul untuk kedalaman tekstur ala 3D render renderan
    return cv2.addWeighted(img, 0.7, emboss, 0.3, 0)

def main():
    image_name, process_mode = get_env_variables()
    
    if not os.path.exists(image_name):
        print(f"[-] Error: File {image_name} tidak ditemukan di root repository!")
        sys.exit(1)
        
    print(f"[+] Membaca gambar: {image_name}")
    img = cv2.imread(image_name)
    
    print(f"[+] Memproses menggunakan mode: {process_mode}")
    
    if process_mode == "upscale":
        out = process_upscale(img)
    elif process_mode == "image vector hitam putih":
        out = process_vector_bw(img)
    elif process_mode == "cartoon style":
        out = process_cartoon(img)
    elif process_mode == "3D style":
        out = process_3d_style(img)
    else:
        print("[-] Mode tidak dikenali.")
        sys.exit(1)
        
    output_path = f"output_{process_mode.replace(' ', '_')}.png"
    cv2.imwrite(output_path, out)
    print(f"[+] Sukses! Hasil disimpan di {output_path}")

if __name__ == "__main__":
    main()
