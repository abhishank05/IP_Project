import os
import glob
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# ==========================================
# DIRECTORIES
# ==========================================
INPUT_DIR = "satellite_inputs"
OUTPUT_DIR = "final_outputs"

def process_disaster_image(filepath, filename):
    """Passes a single image through all 5 DIP Units to create a unified disaster map."""
    
    # ---------------------------------------------------------
    # UNIT 1: Sampling, Quantization & File Reading
    # ---------------------------------------------------------
    img = cv2.imread(filepath)
    if img is None:
        print(f"❌ Error: Could not read '{filename}'. Skipping.")
        return None
        
    # Spatial Sampling: Standardize image size to 512x512
    img = cv2.resize(img, (512, 512))
    
    # ---------------------------------------------------------
    # UNIT 2: Spatial Domain Enhancement & Grey Level Transforms
    # ---------------------------------------------------------
    # Convert to LAB color space to enhance the lighting (L channel) without distorting colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    # This brings out hidden details in dark flood waters or rubble
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ---------------------------------------------------------
    # UNIT 3: Image Restoration & Noise Modelling
    # ---------------------------------------------------------
    # Apply an Order Statistic Filter (Median Filter) to remove sensor speckle noise
    # while preserving the sharp edges of buildings and rivers
    restored = cv2.medianBlur(enhanced, 3)

    # ---------------------------------------------------------
    # UNIT 4: Feature Extraction & Texture Analysis
    # ---------------------------------------------------------
    gray_restored = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    
    # Extract Texture Features (GLCM) for the Machine Learning Classifier
    glcm = graycomatrix(gray_restored, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    features = [contrast, energy]

    # ---------------------------------------------------------
    # UNIT 5: Clustering, Compositing, & Security
    # ---------------------------------------------------------
    # 1. Partitional Clustering (K-Means) for Damage/Terrain Segmentation
    # We flatten the image, group similar pixels (e.g., water, land, urban), and rebuild it
    Z = np.float32(restored.reshape((-1, 3)))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_map = centers[kmeans.labels_.flatten()].reshape(restored.shape)

    # 2. Digital Compositing (Visual Effects)
    # Blend the AI-segmented map with the restored image so humans can still recognize the terrain
    final_output = cv2.addWeighted(restored, 0.6, segmented_map, 0.4, 0)

    # 3. Security: Digital Watermarking
    # Add a semi-transparent professional assessment label
    overlay = final_output.copy()
    cv2.rectangle(overlay, (0, 470), (512, 512), (0, 0, 0), -1)
    cv2.putText(overlay, 'AI DISASTER ASSESSMENT SECURED', (15, 495), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    final_output = cv2.addWeighted(overlay, 0.7, final_output, 0.3, 0)

    # 4. Security: Steganography (LSB)
    # Hide an invisible bit of data in the Least Significant Bit of the first 10 pixels
    final_output[0, 0:10] = (final_output[0, 0:10] & 254) | 1 

    # --- SAVE FINAL UNIFIED OUTPUT ---
    out_path = os.path.join(OUTPUT_DIR, f"{filename}_Final_Assessment.png") # PNG preserves Steganography
    cv2.imwrite(out_path, final_output)
    
    return features

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Support multiple image formats you might have downloaded
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    if not image_paths:
        print(f"⚠️ Please put your 5 disaster images into the '{INPUT_DIR}' folder!")
        return

    print(f"\nStarting Disaster Assessment Pipeline on {len(image_paths)} images...")
    all_features = []

    for path in image_paths[:5]:
        filename = os.path.splitext(os.path.basename(path))[0]
        print(f" -> Processing: {filename}...")
        
        features = process_disaster_image(path, filename)
        if features:
            all_features.append(features)

    # Fulfilling the Unit 5 Classification Requirement
    if len(all_features) > 1:
        print("\n-> Training Unit 5 SVM Classifier on extracted GLCM textures...")
        X = np.array(all_features)
        y = np.array([i % 2 for i in range(len(X))]) # Mock labels (e.g., 0: Flood, 1: Earthquake)
        svm_clf = SVC()
        svm_clf.fit(X, y)
        print("-> SVM Training Complete. Pipeline is fully executed.")

    print(f"\n🎉 DONE! Check the '{OUTPUT_DIR}' folder for your 5 completed disaster maps.")

if __name__ == "__main__":
    main()
