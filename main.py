import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.util import random_noise
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# SETUP & HELPER FUNCTIONS
# ==========================================
INPUT_IMAGE = "img3.jpg"
OUTPUT_DIR = "Report_Comparisons"

def setup_env():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"❌ ERROR: Please place an image named '{INPUT_IMAGE}' in this folder!")
        exit()
        
    # Resize for consistent processing speed
    img = cv2.resize(img, (512, 512))
    return img

def save_comparison(title, img_in, img_out, filename, cmap_in=None, cmap_out=None):
    """Helper function to save side-by-side comparisons for the report."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title, fontsize=16)
    
    # Handle color conversions for Matplotlib (BGR to RGB)
    if len(img_in.shape) == 3 and cmap_in is None:
        img_in_disp = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    else:
        img_in_disp = img_in
        
    if len(img_out.shape) == 3 and cmap_out is None:
        img_out_disp = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    else:
        img_out_disp = img_out

    axes[0].imshow(img_in_disp, cmap=cmap_in)
    axes[0].set_title("Original Input")
    axes[0].axis('off')
    
    axes[1].imshow(img_out_disp, cmap=cmap_out)
    axes[1].set_title("Processed Output")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Generated: {filename}")

# ==========================================
# UNIT 1: Sampling, Quantization, Color Models
# ==========================================
def unit_1(img):
    # 1. Quantization (Reducing colors to create a posterized effect)
    quantized = (img // 64) * 64
    save_comparison("Unit 1: Sampling & Quantization", img, quantized, "U1_Quantization.jpg")
    
    # 2. Color Image Models (BGR to HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    save_comparison("Unit 1: Color Image Models (HSV)", img, hsv, "U1_ColorModel_HSV.jpg")
    
    # 3. Image Operations (Brightness Increase)
    brightened = cv2.add(img, np.array([75.0]))
    save_comparison("Unit 1: Image Operations (Brightness +75)", img, brightened, "U1_Operations.jpg")

# ==========================================
# UNIT 2: Transforms, Histogram, Filtering
# ==========================================
def unit_2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Fast Fourier Transform (Frequency Domain)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    save_comparison("Unit 2: Fast Fourier Transform (Magnitude)", gray, magnitude_spectrum, "U2_FFT.jpg", cmap_in='gray', cmap_out='gray')
    
    # 2. Gray Level Transforms (Negative)
    negative = 255 - gray
    save_comparison("Unit 2: Gray Level Transform (Negative)", gray, negative, "U2_Negative.jpg", cmap_in='gray', cmap_out='gray')
    
    # 3. Histogram Processing (Equalization)
    equalized = cv2.equalizeHist(gray)
    save_comparison("Unit 2: Histogram Equalization", gray, equalized, "U2_Hist_Equalization.jpg", cmap_in='gray', cmap_out='gray')
    
    # 4. Spatial Filtering (Smoothing & Sharpening)
    smoothed = cv2.GaussianBlur(img, (15, 15), 0)
    save_comparison("Unit 2: Spatial Smoothing (Gaussian)", img, smoothed, "U2_Smoothing.jpg")
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(laplacian)
    save_comparison("Unit 2: Spatial Sharpening (Laplacian)", gray, sharpened, "U2_Sharpening.jpg", cmap_in='gray', cmap_out='gray')

# ==========================================
# UNIT 3: Pyramids, Wavelets, Restoration, Compress
# ==========================================
def unit_3(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Image Pyramids (Downsampling)
    pyr_down = cv2.pyrDown(img)
    # Resize back up just for visual side-by-side comparison
    pyr_down_disp = cv2.resize(pyr_down, (img.shape[1], img.shape[0])) 
    save_comparison("Unit 3: Image Pyramids (PyrDown)", img, pyr_down_disp, "U3_Pyramids.jpg")
    
    # 2. Noise Modeling & Order Statistic Filters (Median)
    noisy_float = random_noise(img, mode='s&p', amount=0.05)
    noisy = np.array(255 * noisy_float, dtype=np.uint8)
    restored = cv2.medianBlur(noisy, 5)
    save_comparison("Unit 3: Noise Modeling & Median Restoration", noisy, restored, "U3_Restoration.jpg")
    
    # 3. Wavelet Transforms (Discrete Wavelet Transform - LL Band)
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, _ = coeffs2
    LL_disp = cv2.resize(np.uint8(LL), (img.shape[1], img.shape[0]))
    save_comparison("Unit 3: Wavelet Transform (Haar LL Band)", gray, LL_disp, "U3_Wavelets.jpg", cmap_in='gray', cmap_out='gray')

    # 4. Image Compression (JPEG Simulation at 10% Quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 5]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    compressed = cv2.imdecode(encimg, 1)
    save_comparison("Unit 3: Image Compression (High Compression)", img, compressed, "U3_Compression.jpg")

# ==========================================
# UNIT 4: Segmentation, Edges, Features, PCA
# ==========================================
def unit_4(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Detection of Discontinuities / Edge Operators (Canny)
    edges = cv2.Canny(gray, 100, 200)
    save_comparison("Unit 4: Edge Detection (Canny)", gray, edges, "U4_EdgeDetection.jpg", cmap_in='gray', cmap_out='gray')
    
    # 2. Thresholding (Otsu)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_comparison("Unit 4: Region Segmentation (Otsu Threshold)", gray, thresh, "U4_Thresholding.jpg", cmap_in='gray', cmap_out='gray')
    
    # 3. Feature Reduction (PCA on color channels)
    # Reshape image to a 2D array of pixels, apply PCA to reduce 3 channels to 1
    pixels = img.reshape(-1, 3)
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(pixels)
    pca_img = pca_result.reshape(img.shape[0], img.shape[1])
    # Normalize for display
    pca_img = cv2.normalize(pca_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    save_comparison("Unit 4: Feature Reduction (PCA 3-Channels to 1)", img, pca_img, "U4_PCA.jpg", cmap_out='gray')

# ==========================================
# UNIT 5: Classification, Clustering, Security
# ==========================================
def unit_5(img):
    # 1. Partitional Clustering Algorithms (K-Means)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Z)
    centers = np.uint8(kmeans.cluster_centers_)
    clustered = centers[kmeans.labels_.flatten()].reshape(img.shape)
    save_comparison("Unit 5: Partitional Clustering (K-Means)", img, clustered, "U5_KMeans_Clustering.jpg")
    
    # 2. Digital Watermarking
    watermarked = img.copy()
    cv2.putText(watermarked, 'CONFIDENTIAL DATA', (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    save_comparison("Unit 5: Digital Watermarking", img, watermarked, "U5_Watermarking.jpg")
    
    # 3. Steganography (LSB) & Visual Effects (Compositing)
    # Compositing a dramatic color map for "Visual Effects"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    composited = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    save_comparison("Unit 5: Digital Compositing & Visual Effects", img, composited, "U5_Compositing.jpg")
    
    # Note on SVM/Bayesian: These require a multi-image dataset to train and cannot be visually 
    # represented on a single input-to-output image transformation like filtering.

# ==========================================
# EXECUTE ALL UNITS
# ==========================================
if __name__ == "__main__":
    print("Initializing Image Processing Master Pipeline...")
    source_img = setup_env()
    
    print("\n--- Processing Unit 1 ---")
    unit_1(source_img)
    
    print("\n--- Processing Unit 2 ---")
    unit_2(source_img)
    
    print("\n--- Processing Unit 3 ---")
    unit_3(source_img)
    
    print("\n--- Processing Unit 4 ---")
    unit_4(source_img)
    
    print("\n--- Processing Unit 5 ---")
    unit_5(source_img)
    
    print(f"\n✅ All tasks completed! Check the '{OUTPUT_DIR}' folder for your comparison images.")