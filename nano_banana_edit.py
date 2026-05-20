import os
from PIL import Image

input_dir = "images"
output_dir = "output_bw_style"
os.makedirs(output_dir, exist_ok=True)

# Lu bisa ganti angka ini (misal 2, 3, atau 4) buat nentuin seberapa besar upscalenya
UPSCALE_FACTOR = 2 

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        print(f"🔧 Memproses {filename} secara lokal...")
        img_path = os.path.join(input_dir, filename)
        
        try:
            # Buka gambar
            img = Image.open(img_path)
            
            # 1. Upscale gambar biar resolusinya naik (Pake LANCZOS biar hasilnya tajam)
            new_size = (int(img.width * UPSCALE_FACTOR), int(img.height * UPSCALE_FACTOR))
            img_upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 2. Ubah ke Grayscale (Abu-abu)
            img_gray = img_upscaled.convert('L')
            
            # 3. Terapkan Thresholding biar jadi Hitam Putih Solid (Vector Aesthetic)
            # Pixel yang lebih terang dari 128 jadi putih, sisanya jadi hitam murni
            threshold = 128
            img_bw = img_gray.point(lambda p: 255 if p > threshold else 0)
            
            # Ubah format ke 1-bit pixel (murni hitam putih)
            img_bw = img_bw.convert('1') 
            
            # Simpan hasil output raster
            output_path = os.path.join(output_dir, f"bw_{filename}")
            img_bw.save(output_path)
            print(f"✅ Sukses: {output_path}")
            
        except Exception as e:
            print(f"❌ Gagal memproses {filename}: {e}")
