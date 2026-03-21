# Satellite Image Processing Pipeline

An end-to-end Python pipeline for processing, enhancing, and securing raw satellite imagery using OpenCV, Scikit-Image, and Scikit-Learn.

## Overview
This project takes raw, high-depth (16-bit) satellite `.tif` files and pushes them through a 5-unit Digital Image Processing (DIP) pipeline to produce clean, color-corrected, and segmented geographic maps.

### Pipeline Units:
1. **Sampling & Quantization:** Robust ingestion of scientific TIFF data and 8-bit normalization.
2. **Image Enhancement:** Contrast Limited Adaptive Histogram Equalization (CLAHE) to fix sensor lighting issues.
3. **Image Restoration:** Spatial median filtering to remove high-frequency sensor noise.
4. **Feature Extraction:** Edge detection (Canny) and Texture analysis (GLCM Contrast & Energy).
5. **Machine Learning & Security:** - K-Means Clustering for terrain segmentation.
   - Support Vector Machine (SVM) training on extracted textures.
   - Digital Watermarking and LSB Steganography for data security.

## Requirements
```bash
pip install opencv-python numpy tifffile scikit-image scikit-learn scipy pywavelets