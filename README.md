# ✈️ Flight Delay & Cancellation Prediction (2019–2023)

![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Project-Complete-success)

Welcome to a deep dive into U.S. flight data — using GPU-accelerated machine learning to predict delays and cancellations across millions of flights. This project analyzes five years of FAA data (2019–2023) and brings together data science, real-world insights, and raw GPU power.

✨ Powered by NVIDIA RAPIDS (`cuML`), `XGBoost`, and real U.S. flight data.


---

## 🚀 Project Highlights

- **Dataset**: [Flight Delay and Cancellation (2019–2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
- **Languages/Tools**: Python, Pandas, cuDF, XGBoost, cuML (GPU-accelerated), scikit-learn
- **Notebook**: [flight-delay-prediction.ipynb](flight-delay-prediction.ipynb)
- **GPU Utilization**: NVIDIA Tesla T4 (via Google Colab)
- **Average Training Time**: ~3.2× speedup using GPU vs CPU
- **Model Accuracy**: Up to **88.3%** using GPU-based XGBoost on balanced data

---

## 📊 Exploratory Data Analysis (EDA)

We explored key variables influencing delays:
- ✈️ **Airline and Route**: Some carriers/airports have significantly more delays.
- 🕰️ **Scheduled Times**: Rush hours and red-eye slots tell interesting stories.
- 🗓️ **Weekday/Weekend Trends**: Friday flights? Good luck. 😅

---

## 🧪 Modeling

We trained multiple models using both **CPU** and **GPU** pipelines:

| Model                  | Type | Accuracy | Notes |
|------------------------|------|----------|-------|
| Logistic Regression    | GPU  | 81.2%    | Fast, great baseline |
| Random Forest          | GPU  | 85.7%    | Handles non-linear patterns well |
| Support Vector Machine | GPU  | 82.4%    | Sensitive to scaling |
| XGBoost                | GPU  | **88.3%**| Top performer |
| XGBoost                | CPU  | 82.1%    | Slower and less accurate |

✅ **GPU reduced training time by ~3x**, especially for ensemble models.

---

## ⏱️ GPU Time & Performance

With NVIDIA T4 via Colab:
- XGBoost training time: **~6.1s (GPU)** vs **~18.9s (CPU)**
- cuML models showed **consistent speedup** over their scikit-learn counterparts
- `GPUtil` used to verify memory and GPU utilization

---

## 📁 File Structure

```
📦flight-delay-prediction
 ┣ 📓 yousif.ipynb              ← Main analysis notebook
 ┣ 📁 data/                     ← Contains flight data CSVs
 ┣ 📄 README.md                 ← This file!
 ┗ 📄 requirements.txt          ← Package dependencies
```

---

## 🔧 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/flight-delay-prediction.git
   cd flight-delay-prediction
   ```

2. Set up dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open notebook:
   ```bash
   jupyter notebook yousif.ipynb
   ```

> ✅ Note: To benefit from GPU acceleration, use Google Colab or a local CUDA-enabled environment.

---

## 📌 Takeaways

- GPU acceleration can **significantly reduce training time** and **increase scalability**.
- Airline, airport, and schedule patterns play a huge role in delays.
- Even basic feature engineering (like time of day or airport codes) can boost predictive power.

---

## 📬 Contact

Made with 💻, ✈️, and a sprinkle of delay frustration.  
If you like it, ⭐️ the repo or reach out!
