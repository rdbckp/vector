import os
import sys
import cv2
import numpy as np
import requests

def get_env_variables():
    image_source = os.environ.get("IMAGE_SOURCE", "").strip()
    process_mode = os.environ.get("PROCESS_MODE", "upscale")
    return image_source, process_mode

def load_image(source):
    # Cek apakah input berupa URL
    if source.startswith("http://") or source.startswith("https://"):
        print(f"[+] Mendownload gambar dari URL: {source}")
        try:
            response = requests.get(source, timeout=15)
            response.raise_for_status()
            # Ubah bytes data menjadi numpy array untuk dibaca OpenCV
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                print("[-] Error: URL valid tapi data bukan format gambar yang didukung.")
                sys.exit(1)
            return img
        except Exception as e:
            print(f"[-] Gagal mendownload gambar. Error: {e}")
            sys.exit(1)
    else:
        # Jika bukan URL, anggap sebagai file lokal di root repo
        if not os.path.exists(source):
            print(f"[-] Error: File lokal '{source}' tidak ditemukan di root repo!")
            sys.exit(1)
        print(f"[+] Membaca file lokal: {source}")
        return cv2.imread(source)

def process_upscale(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

def process_vector_bw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    return bw

def process_cartoon(img):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 9, 2)
    return cv2.bitwise_and(color, color, mask=edges)

def process_3d_style(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    emboss_kernel = np.array([[ -2, -1, 0],
                              [ -1,  1, 1],
                              [  0,  1, 2]])
    emboss = cv2.filter2D(blur, -1, emboss_kernel)
    return cv2.addWeighted(img, 0.7, emboss, 0.3, 0)

def main():
    image_source, process_mode = get_env_variables()
    
    if not image_source:
        print("[-] Error: Input Image Source kosong!")
        sys.exit(1)
        
    img = load_image(image_source)
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
