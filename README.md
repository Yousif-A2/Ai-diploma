# ✈️ Flight Delay & Cancellation Predictor (2019–2023)

![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Project-Complete-success)

Welcome to a high-flying journey into the world of U.S. air travel, where we combine data science, machine learning, and GPU acceleration to predict flight delays and cancellations. This project explores over **3 million flight records** from **2019 to 2023**, aiming to identify patterns, bottlenecks, and predictors of delayed flights.

> ✨ Powered by NVIDIA RAPIDS (`cuML`), `XGBoost`, and real U.S. flight data.

---

## 🚀 Project Highlights

- 📊 **Dataset:** 3M+ records of U.S. domestic flights (2019–2023) from [Kaggle](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
- 🧠 **Models Used:** 
  - cuML Random Forest
  - cuML Logistic Regression
  - cuML Support Vector Classifier
  - XGBoost (for comparison)
- ⚡ **Accelerated Computing:** GPU-powered training with RAPIDS (cuML) + CUDA
- 📈 **Evaluation Metrics:** Accuracy, Classification Report, Confusion Matrix

---

## 🏁 Results Snapshot

| Model                   | Accuracy |
|------------------------|----------|
| 🧠 **cuML Random Forest** | ✅ **95.7%** |
| 🧠 cuML SVC             | 92.3%    |
| ⚙️ XGBoost              | 91.8%    |

- 🚀 **GPU Detected:** Yes (via `GPUtil`)
- 🕒 **Speed Boost:** Training time reduced by ~4x using GPU over CPU
- 📌 **Feature Engineering:** Includes time slots, distances, carrier IDs
- 🔍 **Key Insight:** Flights between 6 PM–11 PM showed higher delay probabilities

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction

pip install -r requirements.txt
