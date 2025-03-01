# **Face Recognition Using Dlib and SVM Classifier**


This project implements a **face recognition system** using **Dlib** for face detection, landmark extraction, and encoding, with an **SVM classifier** to classify known faces. The system aligns faces, extracts feature encodings, and then predicts the identity of a person in an image.

---

## **How It Works**
1. **Face Detection:**  
   - Uses Dlib's `get_frontal_face_detector()` to detect faces in an image.
  
2. **Face Alignment:**  
   - Extracts **68 facial landmarks** using `shape_predictor_68_face_landmarks.dat`.
   - Aligns the face.

3. **Feature Extraction (Encoding):**  
   - Converts each aligned face into a **128-dimensional feature vector** using `dlib.face_recognition_model_v1()`.

4. **Training the SVM Classifier:**  
   - Collects multiple face encodings from training images.
   - Labels them and fits an **SVM classifier** (`SVC(kernel='linear', probability=True)`).
   - The classifier learns to differentiate between characters.

5. **Face Recognition (Prediction):**  
   - Detects a face in a new image.
   - Aligns and extracts features.
   - Uses the trained **SVM model** to predict the person's identity.

---

## **Files**
- **`shape_predictor_68_face_landmarks.dat`**  Dlib model for detecting facial landmarks.
- **`dlib_face_recognition_resnet_model_v1.dat`**  Pre-trained Dlib model for generating 128D face encodings.
- **`face_recognition.ipynb`**  Processes images, extracts face encodings, and trains the SVM model and KNN classifier model, detects a face in an image, and predicts the identity.

---

## **Setup & Installation**
1. **Install dependencies:**
   ```bash
   pip install numpy opencv-python dlib matplotlib pandas scikit-learn
   ```
2. **Ensure you have Dlib’s pre-trained models** (`.dat` files) in your project directory.
3. **Run:**
   ```bash
   face_recognition.ipynb
   ```

---

### **Next Steps**
- Extend to **real-time face recognition** using a webcam.
- Improve accuracy by **adding more training images**.

---
