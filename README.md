# 💧 Water Potability Prediction

This project predicts whether a given water sample is safe for human consumption based on its physicochemical properties. Using machine learning models, the app helps determine potability, which is crucial for ensuring safe drinking water.

---

## 🔍 Problem Statement

Contaminated water is a major global issue. The goal of this project is to predict the potability of water using features like pH, hardness, solids, and more, enabling early identification of unsafe water sources.

---

## 📊 Dataset Information

- Source: [Kaggle - Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- Total Records: 3,276
- Target: `Potability` (1 = Safe, 0 = Not Safe)
- Features:
  - `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`
  - `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`

---

## ⚙️ Tech Stack

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Deployment**: Streamlit

---

## 🚀 How to Run Locally

1. Clone the repository  
```bash
git clone https://github.com/73hr4774/Water_potability_prediction.git
cd Water_potability_prediction

pip install -r requirements.txt

streamlit run app.py
🌐 Deployed App
👉 Click here to view the deployed Streamlit app
https://waterpotabilityprediction-app.streamlit.app/

🧠 Key Insights
Water with low pH and high levels of solids or trihalomethanes tends to be non-potable.

Random Forest performed best in terms of classification accuracy.

Feature importance analysis helped identify major contributors to water potability.

