import os
import sys
import cv2
import numpy as np
import requests
from PIL import Image
import io
from super_image import EdsrModel, ImageLoader

def get_env_variables():
    image_source = os.environ.get("IMAGE_SOURCE", "").strip()
    process_mode = os.environ.get("PROCESS_MODE", "upscale")
    return image_source, process_mode

def load_image_pil(source):
    if source.startswith("http://") or source.startswith("https://"):
        print(f"[+] Mendownload gambar dari URL: {source}")
        try:
            response = requests.get(source, timeout=15)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"[-] Gagal mendownload gambar. Error: {e}")
            sys.exit(1)
    else:
        if not os.path.exists(source):
            print(f"[-] Error: File lokal '{source}' tidak ditemukan!")
            sys.exit(1)
        print(f"[+] Membaca file lokal: {source}")
        return Image.open(source).convert("RGB")

def process_upscale_ai(pil_img):
    print("[+] Menjalankan Deep Learning AI (EDSR Model x2) - Anti Burik...")
    try:
        # Load pre-trained model AI khusus penanganan restorasi gambar hancur
        model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
        inputs = ImageLoader.load_image(pil_img)
        preds = model(inputs)
        
        # Simpan hasil prediksi tensor AI ke PIL Image
        output_dir = "./"
        ImageLoader.save_image(preds, os.path.join(output_dir, "output_upscale.png"))
        print("[+] AI Sukses merekonstruksi gambar!")
        return True
    except Exception as e:
        print(f"[-] AI gagal berjalan: {e}. Fallback ke algoritma filter kontras...")
        return False

def process_vector_bw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return bw

def process_cartoon(img):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    return cv2.bitwise_and(color, color, mask=edges)

def process_3d_style(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    emboss_kernel = np.array([[ -2, -1, 0], [ -1,  1, 1], [  0,  1, 2]])
    emboss = cv2.filter2D(blur, -1, emboss_kernel)
    return cv2.addWeighted(img, 0.7, emboss, 0.3, 0)

def main():
    image_source, process_mode = get_env_variables()
    if not image_source:
        print("[-] Error: Input Image Source kosong!")
        sys.exit(1)
        
    pil_img = load_image_pil(image_source)
    print(f"[+] Memproses menggunakan mode: {process_mode}")
    
    output_path = f"output_{process_mode.replace(' ', '_')}.png"
    
    if process_mode == "upscale":
        success = process_upscale_ai(pil_img)
        if not success:
            # Fallback jika model crash (ubah PIL ke OpenCV)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            out = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(output_path, out)
    else:
        # Konversi PIL ke format matriks OpenCV untuk mode efek lainnya
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if process_mode == "vector_bw":
            out = process_vector_bw(img)
        elif process_mode == "cartoon":
            out = process_cartoon(img)
        elif process_mode == "style_3d":
            out = process_3d_style(img)
        else:
            print(f"[-] Mode '{process_mode}' tidak dikenali.")
            sys.exit(1)
        cv2.imwrite(output_path, out)
        
    print(f"[+] Selesai! Hasil akhir diproses.")

if __name__ == "__main__":
    main()
