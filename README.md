# 📊 COVID-19 Trend Analysis & Prediction

Welcome to **COVID-19 Trend Analysis & Prediction** 🚀! This project is designed to analyze COVID-19 case trends, predict future cases using ML models, and visualize data using an interactive **Streamlit Dashboard**.

🔗 **GitHub Repository**: [COVID-19 Trend Analysis & Prediction](https://github.com/debanganghosh08/COVID_19_Trend_Analysis_-_Prediction.git)

---

## 📌 Project Overview

This project focuses on:
✅ Analyzing COVID-19 cases using **real-world datasets** 📊  
✅ **Enhancing data accuracy** using external datasets 🔄  
✅ Building **ML models** (Linear Regression, ARIMA, LSTM) to predict future cases 🧠  
✅ **Interactive Dashboard** using Streamlit for visualization 🎨  
✅ Making the analysis **colorful, complex, and professional** 💡  

---

## 🔥 Tech Stack & Concepts Used

### 🛠️ Tools & Libraries
- **Pandas, NumPy** → Data Processing & Cleaning
- **Matplotlib, Seaborn** → Data Visualization
- **Scikit-learn** → Machine Learning (Linear Regression)
- **Statsmodels** → ARIMA (Time-series analysis)
- **TensorFlow/Keras** → LSTM (Deep Learning)
- **Streamlit** → Interactive Dashboard

### 📖 Key Concepts
- Data Cleaning & Preprocessing 🧹
- Time-series Analysis 📈
- Machine Learning & Deep Learning 🤖
- Feature Engineering 🏗️
- Dashboard Integration 💻

---

## 🚀 Step-by-Step Implementation

### 🔹 Step 1: Dataset Loading & Preprocessing

We used **Kaggle's COVID-19 dataset** and enhanced its accuracy using an **external dataset** containing daily case reports.

```python
import pandas as pd

# Load Main Dataset
covid_df = pd.read_csv("meirnizri/covid19-dataset.csv")

# Load External Dataset
external_df = pd.read_csv("external_dataset.csv")

# Convert Date Columns to DateTime Format
covid_df['DATE_DIED'] = pd.to_datetime(covid_df['DATE_DIED'], errors='coerce')
external_df['Date'] = pd.to_datetime(external_df['Date'])
```

**🔹 Enhancements Made:**
- Fixed column mismatches 🔄
- Standardized date formats 🗓️
- Merged datasets for better accuracy 📊

---

### 🔹 Step 2: Data Visualization (Professional & Colorful)

Created **colorful and complex visualizations** to analyze case trends and relationships.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
sns.lineplot(x=external_df['Date'], y=external_df['Confirmed'], label='Confirmed Cases', color='blue')
sns.lineplot(x=external_df['Date'], y=external_df['Deaths'], label='Deaths', color='red')
sns.lineplot(x=external_df['Date'], y=external_df['Recovered'], label='Recovered', color='green')
plt.xlabel("Date")
plt.ylabel("Case Count")
plt.title("COVID-19 Cases Over Time")
plt.legend()
plt.show()
```

**🔹 Improvements Made:**
- Used **Seaborn for complex visuals** 🌈
- Improved readability & aesthetics 📊

---

### 🔹 Step 3: Machine Learning Models for Prediction

We implemented **three ML models** to predict future COVID-19 cases:
1️⃣ **Linear Regression** – Simple and effective 📉  
2️⃣ **ARIMA** – Best for short-term forecasting 📊  
3️⃣ **LSTM** – Captures long-term dependencies 🧠  

#### 📌 Linear Regression Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare Data for ML Model
X = df[['Days']]
y = df['Confirmed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predict Future Cases
future_days = np.array(range(X.max()[0] + 1, X.max()[0] + 30)).reshape(-1, 1)
future_predictions = model_lr.predict(future_days)
```

**🔹 Changes & Enhancements:**
- Used **feature engineering** (Days instead of Date) 🏗️
- Implemented **ARIMA & LSTM** for better forecasting 🧠

---

### 🔹 Step 4: Streamlit Dashboard Integration

An interactive **Streamlit Dashboard** was created to display real-time COVID-19 trends and predictions.

```python
import streamlit as st

st.title("COVID-19 Trend Analysis & Prediction Dashboard")

# Sidebar Navigation
view = st.sidebar.selectbox("Select View", ["Overview", "Predictions"])

if view == "Overview":
    st.write("### COVID-19 Cases Over Time")
    st.line_chart(df[['Date', 'Confirmed', 'Deaths', 'Recovered']].set_index('Date'))
```

🔹 **Features Added:**
- **Dynamic case updates** 📊
- **Sidebar for better navigation** 🗂️
- **Real-time trend visualization** 🎨

---

## 🏆 Final Results
✅ **Data Successfully Cleaned & Merged** 🧹  
✅ **Professional & Interactive Visualizations** 🎨  
✅ **3 Machine Learning Models Implemented** 🤖  
✅ **Interactive Dashboard Integrated** 🚀  

---

## 🚀 Running the Project Locally

### 🔹 Install Dependencies
```sh
pip install -r requirements.txt
```

### 🔹 Run the Streamlit Dashboard
```sh
streamlit run dashboard.py
```

---

## 🌍 Future Enhancements
🔹 Deploy the project on **Streamlit Cloud or AWS** ☁️  
🔹 Optimize ML models for **better accuracy** 🎯  
🔹 Add **Geospatial Analysis** for region-wise predictions 🌎  

📌 **Want to contribute?** Feel free to fork the repository and make improvements! 🚀

---

## ❤️ Contributors
👤 **[Your Name]** – Developer & Data Scientist 🚀

📩 **Contact:** your-email@example.com

---

⭐ **If you found this project useful, please give it a star!** ⭐

