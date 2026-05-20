import os
from google import genai
from PIL import Image

# Ngambil API Key dari GitHub Secrets
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Bro! API Key belum diset. Masukin GEMINI_API_KEY di GitHub Secrets.")

client = genai.Client(api_key=api_key)

input_dir = "images"
output_dir = "output_bw_style"
os.makedirs(output_dir, exist_ok=True)

# Prompt ketat biar Nano-Banana fokus nge-upscale & ganti style, BUKAN ngarang elemen baru
prompt_text = "Convert this image into a crisp, high-resolution, black and white vector-style graphic. Upscale the quality but DO NOT change the core graphics, shapes, layouts, or original subjects. Just apply the clean monochrome vector aesthetic."

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        print(f"🍌 Memproses {filename} dengan Nano-Banana...")
        img_path = os.path.join(input_dir, filename)
        
        try:
            img = Image.open(img_path)
            
            # Memanggil API Nano-Banana (Gemini 3.1 Flash Image / Imagen)
            result = client.models.generate_images(
                model='gemini-3.1-flash', 
                prompt=prompt_text,
                image=img,
                number_of_images=1,
                output_mime_type="image/png"
            )
            
            # Simpan hasil output raster bergaya vektor
            for i, generated_image in enumerate(result.generated_images):
                output_path = os.path.join(output_dir, f"bw_{filename}")
                generated_image.image.save(output_path)
                print(f"✅ Sukses: {output_path}")
                
        except Exception as e:
            print(f"❌ Gagal memproses {filename}: {e}")
