# Airbnb NYC Price Prediction 🏠💰

This project utilizes **machine learning** to predict **Airbnb rental prices** in **New York City**. By analyzing various factors such as **location, property type, and amenities**, this model helps both **hosts** and **guests** make informed pricing decisions.

---

## 📌 Project Overview

The goal of this project is to **develop a predictive model** that estimates Airbnb listing prices using **data-driven insights**. The model is trained on a dataset that contains **listing details, neighborhood characteristics, and pricing trends**.

By leveraging **regression algorithms**, we analyze pricing patterns and help hosts optimize rental rates, while giving potential guests a better understanding of pricing expectations.

---

## 📊 Dataset Information

The dataset used in this project is sourced from **[Inside Airbnb](http://insideairbnb.com/new-york-city/)**. It includes comprehensive information about Airbnb listings in NYC, such as:
- **Property type and room type**
- **Location (latitude & longitude)**
- **Price per night**
- **Availability and reviews**

### 🔍 Data Attribution

The data from **Inside Airbnb** is provided under an **open data license**. Although specific licensing details are not included, we acknowledge and appreciate their contribution to the **open data community**.

### 📖 Citation

If you use this dataset, please consider citing **Inside Airbnb** as follows:

```
Inside Airbnb. (Year). "NYC Airbnb Dataset," Inside Airbnb. URL: [http://insideairbnb.com/](http://insideairbnb.com/)
```

---

## 🛠️ Setting Up the Project

To run this project locally, follow these steps:

1️⃣ **Clone the repository**
```bash
git clone https://github.com/jibnorge/Airbnb-NYC-Price-Prediction.git
cd Airbnb-NYC-Price-Prediction
```

2️⃣ **Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Run the Jupyter Notebook**
```bash
jupyter notebook
```
Open and execute `airbnb.ipynb` to explore the dataset and train models.

---

## 📊 Data Preparation & Analysis

### 🔹 Data Cleaning
- Identified and handled **missing values, duplicates, and outliers**.
- Standardized **categorical variables** for model training.
- Converted **date columns** for better insights.

📌 **Dataset Shape After Cleaning:** `(48,895 rows × 16 columns)`

### 🔹 Exploratory Data Analysis (EDA)
- **Neighborhood distribution of listings**
  ![](images/map%20final.png)

- **Room type frequency**
  ![](images/neigh_group%20and%20room.png)

- **Price distribution analysis**
  ![](images/distribution%20of%20price.png)

### 🔹 Feature Engineering
- **One-hot encoding** for categorical features.
- **Box-Cox transformations** to normalize skewed distributions.
- **Outlier removal** to improve model performance.

---

## 🤖 Model Training & Evaluation

Several machine learning models were tested to predict Airbnb rental prices:

### **1️⃣ Linear Regression**
✅ **Best Parameters:** `{fit_intercept: True}`  
✅ **Accuracy (R² Score):** `52%`  

### **2️⃣ Decision Tree**
✅ **Best Parameters:** `{max_depth: 5, min_samples_leaf: 2}`  
✅ **Accuracy (R² Score):** `56%`  

### **3️⃣ Random Forest**
✅ **Best Parameters:** `{n_estimators: 130, max_depth: 10}`  
✅ **Accuracy (R² Score):** `62%`  

### **4️⃣ Boosting Models (Best Performance)**
| Model        | Accuracy (R²) |
|-------------|--------------|
| XGBoost     | **64%**      |
| CatBoost    | 63%          |
| LightGBM    | 63%          |
| GradientBoost | 63%        |

🏆 **Best Model: XGBoost** with **64% accuracy**.

![](images/cv%20point%20plot.png)

---

## 🔑 Key Findings

- **XGBoost outperformed all other models** with the highest accuracy.
- **Room type and location** are the most significant factors affecting price.
- **Data transformations** improved model performance.

---

## 📌 Recommendations

🔹 **Feature Selection:** Fine-tuning features could further enhance accuracy.  
🔹 **Hyperparameter Tuning:** More grid searches could improve model efficiency.  
🔹 **Dynamic Pricing:** Applying real-time price updates can optimize revenue.  

---

## 🎯 Conclusion

This project demonstrates the **power of machine learning** in predicting Airbnb rental prices. With further optimizations, this model can be used by **hosts** to **maximize earnings** and **guests** to find the best deals.

For future improvements, **adding real-time data, integrating external datasets, and enhancing feature engineering** could take the model to the next level.

---

## ⭐ Support & Contributions

If this project was useful to you, **consider giving it a ⭐ on GitHub**!  

For suggestions or contributions, **submit a pull request**! 🚀  
