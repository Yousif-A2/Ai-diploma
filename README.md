# ✈️ Flight Delay & Cancellation Predictor (2019–2023)

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
- 🧮 **Accelerated Computing:** GPU acceleration with RAPIDS & CUDA
- 📈 **Evaluation Metrics:** Accuracy, Classification Report, Confusion Matrix

---

## 🏁 Results Snapshot

| Model                   | Accuracy |
|------------------------|----------|
| **cuML Random Forest** | **95.7%** |
| cuML SVC               | 92.3%    |
| XGBoost                | 91.8%    |

- 🚀 **GPU Detected:** Yes (via `GPUtil`)
- 🕒 **Training Time:** ~4x faster on GPU than CPU-based models
- 🔎 **Feature Engineering:** Time of day, distance bands, airline ID, etc.
- 🔍 **Insight:** Flights scheduled between 6 PM–11 PM had a higher delay rate.

---

## 📦 Setup

```bash
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction
pip install -r requirements.txt
