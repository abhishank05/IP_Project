import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.util import random_noise
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================
# SETUP & DIRECTORIES
# ==========================================
INPUT_IMAGE = "img3.jpg"
OUTPUT_DIR = "Report_Comparisons"
MATRIX_DIR = "Report_Matrices"
ANALYSIS_FILE = "Report_Analysis.txt"

def setup_env():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"❌ ERROR: Please place an image named '{INPUT_IMAGE}' in this folder!")
        exit()
        
    # Resize for consistent processing speed
    img = cv2.resize(img, (512, 512))
    return img

# ==========================================
# UTILITY FUNCTIONS (MATRIX & ENTROPY)
# ==========================================
def to_uint8(img):
    """Safely convert float/negative arrays (like FFT/Wavelet) to 8-bit."""
    img = np.abs(img)
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
    return (img * 255).astype(np.uint8)

def entropy(img):
    """Calculate Shannon Entropy for the analysis file."""
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,256))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def save_matrix(img, filepath, label):
    """Downsamples the image to a 10x10 grid to show the mathematical pixel changes."""
    # Convert to grayscale if it's a color image so the matrix is 2D and readable
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Safely convert floating point images to 8-bit integers for text display
    if img.dtype != np.uint8:
        img = to_uint8(img)
        
    small = cv2.resize(img, (10, 10), interpolation=cv2.INTER_AREA)
    
    with open(filepath, "w") as f:
        f.write(f"--- {label.upper()} MATRIX ---\n\n")
        for row in small:
            f.write(" ".join(f"{int(val):3d}" for val in row) + "\n")

def save_matrix_and_analysis(title, img_in, img_out, prefix, step_num):
    """Handles generating both the text matrices and the entropy analysis log."""
    # 1. Save Matrices
    in_path = os.path.join(MATRIX_DIR, f"{prefix}_Input.txt")
    out_path = os.path.join(MATRIX_DIR, f"{prefix}_Output.txt")
    save_matrix(img_in, in_path, f"{title} (Input)")
    save_matrix(img_out, out_path, f"{title} (Output)")
    
    # 2. Log Entropy Analysis & Print to Terminal
    ent_in = entropy(img_in)
    ent_out = entropy(img_out)
    log_string = f"{step_num}. {title.ljust(22)} {ent_in:.2f} -> {ent_out:.2f}"
    
    print(log_string)
    with open(ANALYSIS_FILE, "a", encoding="utf-8") as af:
        af.write(log_string + "\n")

# ==========================================
# VISUAL SAVING FUNCTION
# ==========================================
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

# ==========================================
# UNIT 1: Sampling, Quantization, Color Models
# ==========================================
def unit_1(img):
    # 1. Sampling
    sampled = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    sampled_disp = cv2.resize(sampled, (img.shape[1], img.shape[0])) # Resize up just for visual
    save_comparison("Unit 1: Sampling", img, sampled_disp, "U1_01_Sampling.jpg")
    save_matrix_and_analysis("Sampling", img, sampled, "U1_01_Sampling", 1)

    # 2. Quantization
    quantized = (img // 64) * 64
    save_comparison("Unit 1: Quantization", img, quantized, "U1_02_Quantization.jpg")
    save_matrix_and_analysis("Quantization", img, quantized, "U1_02_Quantization", 2)
    
    # 3. Color Image Models (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    save_comparison("Unit 1: Color Model (HSV)", img, hsv, "U1_03_ColorModel_HSV.jpg")
    save_matrix_and_analysis("Color Model (HSV)", img, hsv, "U1_03_ColorModel_HSV", 3)
    
    # 4. Image Operations (Brightness Increase)
    brightened = cv2.add(img, np.array([75.0]))
    save_comparison("Unit 1: Image Operations", img, brightened, "U1_04_Operations.jpg")
    save_matrix_and_analysis("Image Operations", img, brightened, "U1_04_Operations", 4)

# ==========================================
# UNIT 2: Transforms, Histogram, Filtering
# ==========================================
def unit_2(img, gray):
    # 5. Image Negative
    negative = 255 - gray
    save_comparison("Unit 2: Image Negative", gray, negative, "U2_05_Negative.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("Image Negative", gray, negative, "U2_05_Negative", 5)

    # 6. Histogram Equalization
    equalized = cv2.equalizeHist(gray)
    save_comparison("Unit 2: Histogram Equalization", gray, equalized, "U2_06_Hist_Equalization.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("Hist Equalization", gray, equalized, "U2_06_Hist_Equalization", 6)

    # 7. Gaussian Smoothing
    smoothed = cv2.GaussianBlur(img, (15, 15), 0)
    save_comparison("Unit 2: Gaussian Smoothing", img, smoothed, "U2_07_Smoothing.jpg")
    save_matrix_and_analysis("Gaussian Smoothing", img, smoothed, "U2_07_Smoothing", 7)

    # 8. Laplacian Sharpening
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    save_comparison("Unit 2: Laplacian Sharpen", gray, laplacian, "U2_08_Sharpening.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("Laplacian Sharpen", gray, laplacian, "U2_08_Sharpening", 8)

    # 9. Fast Fourier Transform (Magnitude)
    f = np.fft.fftshift(np.fft.fft2(gray))
    fft_mag = to_uint8(20 * np.log(np.abs(f) + 1))
    save_comparison("Unit 2: FFT Magnitude", gray, fft_mag, "U2_09_FFT.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("FFT Magnitude", gray, fft_mag, "U2_09_FFT", 9)

# ==========================================
# UNIT 3: Pyramids, Wavelets, Restoration, Compress
# ==========================================
def unit_3(img, gray):
    # 10. Image Pyramids
    pyr_down = cv2.pyrDown(img)
    pyr_down_disp = cv2.resize(pyr_down, (img.shape[1], img.shape[0])) 
    save_comparison("Unit 3: Image Pyramids", img, pyr_down_disp, "U3_10_Pyramids.jpg")
    save_matrix_and_analysis("Image Pyramids", img, pyr_down, "U3_10_Pyramids", 10)
    
    # 11. Median Restoration
    noisy = np.array(255 * random_noise(img, mode='s&p', amount=0.05), dtype=np.uint8)
    restored = cv2.medianBlur(noisy, 5)
    save_comparison("Unit 3: Median Restoration", noisy, restored, "U3_11_Restoration.jpg")
    save_matrix_and_analysis("Median Restoration", noisy, restored, "U3_11_Restoration", 11)
    
    # 12. Wavelet Transform (LL Band)
    LL, _ = pywt.dwt2(gray, 'haar')
    wavelet = to_uint8(LL)
    wavelet_disp = cv2.resize(wavelet, (img.shape[1], img.shape[0]))
    save_comparison("Unit 3: Wavelet (LL Band)", gray, wavelet_disp, "U3_12_Wavelet.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("Wavelet (LL Band)", gray, wavelet, "U3_12_Wavelet", 12)

    # 13. JPEG Compression
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
    compressed = cv2.imdecode(encimg, 1)
    save_comparison("Unit 3: JPEG Compression", img, compressed, "U3_13_Compression.jpg")
    save_matrix_and_analysis("JPEG Compression", img, compressed, "U3_13_Compression", 13)

# ==========================================
# UNIT 4: Segmentation, Edges, Features, PCA
# ==========================================
def unit_4(img, gray):
    # 14. Canny Edges
    edges = cv2.Canny(gray, 100, 200)
    save_comparison("Unit 4: Canny Edges", gray, edges, "U4_14_Canny.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("Canny Edges", gray, edges, "U4_14_Canny", 14)
    
    # 15. Otsu Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_comparison("Unit 4: Otsu Threshold", gray, thresh, "U4_15_Otsu.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("Otsu Threshold", gray, thresh, "U4_15_Otsu", 15)

    # 16. Distance Transform
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist_uint8 = to_uint8(dist)
    save_comparison("Unit 4: Distance Transform", gray, dist_uint8, "U4_16_Distance.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("Distance Transform", gray, dist_uint8, "U4_16_Distance", 16)

    # 17. Region Segmentation
    kernel = np.ones((5,5), np.uint8)
    region_seg = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    save_comparison("Unit 4: Region Segment", gray, region_seg, "U4_17_Region.jpg", cmap_in='gray', cmap_out='gray')
    save_matrix_and_analysis("Region Segment", gray, region_seg, "U4_17_Region", 17)

    # 18. PCA Reduction
    pixels = img.reshape(-1, 3)
    pca_img = to_uint8(PCA(n_components=1).fit_transform(pixels).reshape(512, 512))
    save_comparison("Unit 4: PCA Reduction", img, pca_img, "U4_18_PCA.jpg", cmap_out='gray')
    save_matrix_and_analysis("PCA Reduction", img, pca_img, "U4_18_PCA", 18)

# ==========================================
# UNIT 5: Classification, Clustering, Security
# ==========================================
def unit_5(img, gray):
    # 19. K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=5).fit(np.float32(img.reshape((-1, 3))))
    clustered = np.uint8(kmeans.cluster_centers_)[kmeans.labels_.flatten()].reshape(img.shape)
    save_comparison("Unit 5: K-Means Clustering", img, clustered, "U5_19_KMeans.jpg")
    save_matrix_and_analysis("K-Means Clustering", img, clustered, "U5_19_KMeans", 19)
    
    # 20. Watermarking
    watermarked = img.copy()
    cv2.putText(watermarked, 'CONFIDENTIAL', (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    save_comparison("Unit 5: Watermarking", img, watermarked, "U5_20_Watermarking.jpg")
    save_matrix_and_analysis("Watermarking", img, watermarked, "U5_20_Watermarking", 20)

    # 21. Steganography
    stego = img.copy()
    stego[0, 0:10] = (stego[0, 0:10] & 254) | 1
    save_comparison("Unit 5: Steganography", img, stego, "U5_21_Steganography.jpg")
    save_matrix_and_analysis("Steganography", img, stego, "U5_21_Steganography", 21)
    
    # 22. Compositing
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    composited = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    save_comparison("Unit 5: Compositing", img, composited, "U5_22_Compositing.jpg")
    save_matrix_and_analysis("Compositing", img, composited, "U5_22_Compositing", 22)

# ==========================================
# EXECUTE ALL UNITS
# ==========================================
if __name__ == "__main__":
    # Clear the analysis file at the start of a fresh run
    open(ANALYSIS_FILE, "w").close()

    print("Initializing Final Image Processing Pipeline...")
    source_img = setup_env()
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    
    print("="*45)
    print(" 📊 FULL PIPELINE ENTROPY ANALYSIS RESULTS")
    print("="*45)

    with open(ANALYSIS_FILE, "a", encoding="utf-8") as af:
        af.write("="*45 + "\n")
        af.write(" 📊 FULL PIPELINE ENTROPY ANALYSIS RESULTS\n")
        af.write("="*45 + "\n")

    unit_1(source_img)
    unit_2(source_img, source_gray)
    unit_3(source_img, source_gray)
    unit_4(source_img, source_gray)
    unit_5(source_img, source_gray)
    
    print("="*45)
    print("\n✅ All 22 tasks completed successfully!")
    print("-> Check 'Report_Comparisons/' for the 22 visual side-by-side images.")
    print("-> Check 'Report_Matrices/' for the 44 input/output 10x10 text matrices.")
    print(f"-> Check '{ANALYSIS_FILE}' for the complete Entropy log.")