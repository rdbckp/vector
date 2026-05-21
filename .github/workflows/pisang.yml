name: Nano Banana Image Processor

on:
  workflow_dispatch:
    inputs:
      image_source:
        description: 'Masukkan URL Gambar ATAU nama file lokal di root (misal: image.png)'
        required: true
        default: 'https://example.com/sample.jpg'
      process_mode:
        description: 'Pilih model opsi gambar'
        required: true
        default: 'upscale'
        type: choice
        options:
          - upscale
          - image vector hitam putih
          - cartoon style
          - 3D style

permissions:
  contents: write

jobs:
  process-image:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install opencv-python-headless pillow numpy requests

      - name: Run Image Processing
        env:
          IMAGE_SOURCE: ${{ github.event.inputs.image_source }}
          PROCESS_MODE: ${{ github.event.inputs.process_mode }}
        run: |
          python process_image.py

      - name: Commit and Push Output Image
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
          
          git add output_*.png
          
          if git diff --staged --quiet; then
            echo "Tidak ada perubahan gambar yang terdeteksi."
          else
            git commit -m "chore: upload hasil proses ${{ github.event.inputs.process_mode }}"
            git push
          fi
