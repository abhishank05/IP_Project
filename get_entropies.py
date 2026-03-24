import cv2
import numpy as np
import pywt
from skimage.util import random_noise
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

INPUT_IMAGE = "img3.jpg"

def entropy(img):
    """Calculates the Shannon Entropy of an image."""
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,256))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def to_uint8(img):
    """Safely converts arrays to 8-bit for accurate entropy calculation."""
    img = np.abs(img)
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
    return (img * 255).astype(np.uint8)

def main():
    print(f"Loading '{INPUT_IMAGE}' for Statistical Analysis...\n")
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"❌ ERROR: Cannot find '{INPUT_IMAGE}'.")
        return
        
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("="*45)
    print(" 📊 FULL PIPELINE ENTROPY ANALYSIS RESULTS")
    print("="*45)

    # --- UNIT 1 ---
    sampled = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    print(f"1. Sampling:            {entropy(img):.2f} -> {entropy(sampled):.2f}")

    quantized = (img // 64) * 64
    print(f"2. Quantization:        {entropy(img):.2f} -> {entropy(quantized):.2f}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(f"3. Color Model (HSV):   {entropy(img):.2f} -> {entropy(hsv):.2f}")

    brightened = cv2.add(img, np.array([75.0]))
    print(f"4. Image Operations:    {entropy(img):.2f} -> {entropy(brightened):.2f}")

    # --- UNIT 2 ---
    negative = 255 - gray
    print(f"5. Image Negative:      {entropy(gray):.2f} -> {entropy(negative):.2f}")
    
    equalized = cv2.equalizeHist(gray)
    print(f"6. Hist Equalization:   {entropy(gray):.2f} -> {entropy(equalized):.2f}")
    
    smoothed = cv2.GaussianBlur(img, (15, 15), 0)
    print(f"7. Gaussian Smoothing:  {entropy(img):.2f} -> {entropy(smoothed):.2f}")
    
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    print(f"8. Laplacian Sharpen:   {entropy(gray):.2f} -> {entropy(laplacian):.2f}")

    f = np.fft.fftshift(np.fft.fft2(gray))
    fft_mag = to_uint8(20 * np.log(np.abs(f) + 1))
    print(f"9. FFT Magnitude:       {entropy(gray):.2f} -> {entropy(fft_mag):.2f}")

    # --- UNIT 3 ---
    pyr_down = cv2.pyrDown(img)
    print(f"10. Image Pyramids:     {entropy(img):.2f} -> {entropy(pyr_down):.2f}")

    noisy = np.array(255 * random_noise(img, mode='s&p', amount=0.05), dtype=np.uint8)
    restored = cv2.medianBlur(noisy, 5)
    print(f"11. Median Restoration: {entropy(noisy):.2f} -> {entropy(restored):.2f}")

    LL, _ = pywt.dwt2(gray, 'haar')
    wavelet = to_uint8(LL)
    print(f"12. Wavelet (LL Band):  {entropy(gray):.2f} -> {entropy(wavelet):.2f}")

    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
    compressed = cv2.imdecode(encimg, 1)
    print(f"13. JPEG Compression:   {entropy(img):.2f} -> {entropy(compressed):.2f}")

    # --- UNIT 4 ---
    edges = cv2.Canny(gray, 100, 200)
    print(f"14. Canny Edges:        {entropy(gray):.2f} -> {entropy(edges):.2f}")

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"15. Otsu Threshold:     {entropy(gray):.2f} -> {entropy(thresh):.2f}")

    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist_uint8 = to_uint8(dist)
    print(f"16. Distance Transform: {entropy(gray):.2f} -> {entropy(dist_uint8):.2f}")

    kernel = np.ones((5,5), np.uint8)
    region_seg = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    print(f"17. Region Segment:     {entropy(gray):.2f} -> {entropy(region_seg):.2f}")

    pixels = img.reshape(-1, 3)
    pca_img = to_uint8(PCA(n_components=1).fit_transform(pixels).reshape(512, 512))
    print(f"18. PCA Reduction:      {entropy(img):.2f} -> {entropy(pca_img):.2f}")

    # --- UNIT 5 ---
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=5).fit(np.float32(img.reshape((-1, 3))))
    clustered = np.uint8(kmeans.cluster_centers_)[kmeans.labels_.flatten()].reshape(img.shape)
    print(f"19. K-Means Clustering: {entropy(img):.2f} -> {entropy(clustered):.2f}")

    watermarked = img.copy()
    cv2.putText(watermarked, 'CONFIDENTIAL', (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    print(f"20. Watermarking:       {entropy(img):.2f} -> {entropy(watermarked):.2f}")

    stego = img.copy()
    stego[0, 0:10] = (stego[0, 0:10] & 254) | 1
    print(f"21. Steganography:      {entropy(img):.2f} -> {entropy(stego):.2f}")

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    composited = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    print(f"22. Compositing:        {entropy(img):.2f} -> {entropy(composited):.2f}")
    print("="*45)

if __name__ == "__main__":
    main()