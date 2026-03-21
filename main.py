import os
import glob
import cv2
import numpy as np
import tifffile  # Better for scientific satellite images
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# Directories
INPUT_DIR = "satellite_inputs"
OUTPUT_DIR = "final_outputs"

def process_satellite_pipeline(path, filename):
    """Passes a single image through all 5 DIP Units sequentially for a clear output."""
    
    # ---------------------------------------------------------
    # UNIT 1: Sampling & Quantization (Robust TIFF Loading)
    # ---------------------------------------------------------
    img = tifffile.imread(path)
    
    if len(img.shape) == 3 and img.shape[0] in [3, 4, 8, 16]:
        img = np.transpose(img, (1, 2, 0))
        
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    p_lower, p_upper = np.percentile(img, (1, 99))
    img_clipped = np.clip(img, p_lower, p_upper)
    
    img_8u = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    img_bgr = cv2.cvtColor(img_8u, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (512, 512))

    # --- SAVE RAW INPUT FOR COMPARISON ---
    # This gives you the "Before" image for your presentation
    raw_out_path = os.path.join(OUTPUT_DIR, f"{filename}_01_Before_Processing.png")
    cv2.imwrite(raw_out_path, img_bgr)

    # ---------------------------------------------------------
    # UNIT 2: Image Enhancement (CLAHE for crisp details)
    # ---------------------------------------------------------
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    enhanced_img = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

    # ---------------------------------------------------------
    # UNIT 3: Image Restoration
    # ---------------------------------------------------------
    restored_img = cv2.medianBlur(enhanced_img, 3)

    # ---------------------------------------------------------
    # UNIT 4: Feature Extraction
    # ---------------------------------------------------------
    gray_restored = cv2.cvtColor(restored_img, cv2.COLOR_BGR2GRAY)
    
    glcm = graycomatrix(gray_restored, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    features = [contrast, energy]

    # ---------------------------------------------------------
    # UNIT 5: Machine Learning, Clustering, & Security
    # ---------------------------------------------------------
    Z = np.float32(restored_img.reshape((-1, 3)))
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=5).fit(Z)
    _ = kmeans.labels_

    final_output = restored_img.copy()
    cv2.putText(final_output, 'Processed Map Data', (10, 490), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    final_output[0, 0:10] = (final_output[0, 0:10] & 254) | 1 

    # --- SAVE FINAL OUTPUT ---
    # This gives you the "After" image for your presentation
    out_path = os.path.join(OUTPUT_DIR, f"{filename}_02_After_Final_Clear.png")
    cv2.imwrite(out_path, final_output) 
    
    return features

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    
    if not image_paths:
        print(f"Please put your .tif files into the '{INPUT_DIR}' folder!")
        return

    print(f"Starting Clear-Image Pipeline on {len(image_paths)} files...\n")
    all_features = []

    for path in image_paths[:5]:
        filename = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing: {filename}...")
        
        try:
            features = process_satellite_pipeline(path, filename)
            all_features.append(features)
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")

    if all_features:
        print("\nTraining SVM Classifier on Unit 4 features...")
        X = np.array(all_features)
        y = np.array([0, 1, 0, 1, 0])[:len(X)] 
        svm_clf = SVC()
        svm_clf.fit(X, y)
        print("SVM Classifier successfully trained!")

    print(f"\n🎉 PIPELINE COMPLETE! Check '{OUTPUT_DIR}' for your Before and After images.")

if __name__ == "__main__":
    main()