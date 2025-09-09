# Task_6_Nationality-Detection
Develop ML model to predict nationality from image, plus emotion. For Indian: add age, dress color. For US: add age. For African: add dress color. Others: nationality + emotion only. GUI with image preview and output section


### Age Model Link ::


https://drive.google.com/file/d/13BHkxvcynb4pa3zjLLOib6L_arOYzM_L/view?usp=drive_link

### Emotion Model Link ::


https://drive.google.com/file/d/1MMqwTjw_4CrZxRxoQ5YrqwxH4dcPvIl7/view?usp=drive_link





###  Nationality Model Link ::


https://drive.google.com/file/d/1IiRMv5nT9D68JrrMKtjjBdPjo27J7NZJ/view?usp=drive_link


# 😊🌍 Emotion & Ethnicity Detector  

## 📌 Problem Statement  
Detect **emotions** (angry, happy, etc.) and **ethnicity** (White, Black, Asian, Indian, Other) from facial images.  

Applications include:  
- 😀 Sentiment analysis  
- 🌍 Demographic profiling  

This project addresses challenges in **multi-task learning** and **diverse facial features**.  

---

## 📂 Dataset  

- **Sources:**  
  - Emotion → FER-2013 (~28,709 images, 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral)  
  - Ethnicity → UTKFace (~23,705 images with age, gender, ethnicity labels)  

- **Preprocessing:**  
  - FER → Grayscale `48x48`  
  - UTKFace → RGB `128x128`  
  - Both normalized  
  - Ethnicity → 5 classes (0–4)  

- **Classes:**  
  - Emotion → 7  
  - Ethnicity → 5  

- **Size:** ~1GB combined  

---

## 🛠 Methodology  

### 🔹 Data Loading & Preprocessing  
- FER-2013 → loaded with **Keras ImageDataGenerator**  
- UTKFace → custom loading with **OpenCV**  

### 🔹 Model Architectures (CNNs)  
- **Emotion Model:**  
  - Input → `48x48x1` (grayscale)  
  - 3 conv layers → Softmax (7 classes)  

- **Ethnicity Model:**  
  - Input → `128x128x3` (RGB)  
  - 3 conv layers → Softmax (5 classes)  

- **Optimizer:** Adam (`lr=0.001`)  
- **Loss:** Categorical Crossentropy  
- **Metrics:** Accuracy  

### 🔹 Training  
- Train/Test Split → **80/20**  
- Epochs → **10**  
- Batch Size → **32**  

### 🔹 Evaluation  
- ✅ Accuracy per task  
- ✅ Predictions on test images  

---

## ⚙ Tools & Libraries  
- 🧠 TensorFlow/Keras  
- 👁 OpenCV  
- 📊 Pandas  
- 🔢 NumPy  

---

## 📊 Results  

- **Emotion Accuracy:** ~75% (typical for FER-2013 due to noisy labels)  
- **Ethnicity Accuracy:** ~80% (based on UTKFace)  
- **Sample Output:** Correctly predicts emotions like *Angry* and ethnicities on test images  

**Limitations:**  
- FER-2013 → inconsistent labels  
- Ethnicity model → sensitive to lighting & pose  

---

## 🚀 Installation  
```bash
pip install tensorflow opencv-python pandas numpy
