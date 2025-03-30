# 🖨️ Using Regression to Approximate 3D Printing Parameters

## 📌 Project Overview
This project develops a machine learning model to predict the print quality, accuracy, and strength of a 3D printer’s output, helping to optimize 3D printing settings for better performance. The three key target variables are:

✅ **Roughness (μm)** – Measures surface texture and smoothness  
✅ **Elongation (%)** – Indicates material flexibility before failure  
✅ **Tensile Strength (MPa)** – Represents the material’s resistance to tension  

By predicting these metrics, the model aids in refining printer parameters, reducing material waste, and improving manufacturing efficiency.

## 🔍 Approach & Methodology
1️⃣ **Exploratory Data Analysis (EDA)** to identify trends, correlations, and outliers  
2️⃣ **Baseline Linear Regression Models** for each target variable  
3️⃣ **Feature Selection & Engineering** to improve prediction accuracy  
4️⃣ **Model Optimization** using advanced regression techniques  

- **Roughness**: Optimized using **XGBoost** for better generalization  
- **Elongation & Tensile Strength**: Optimized using **Ridge & Lasso Regression**  

## 🔬 Key Observations & Conclusions
- **Roughness** had the strongest correlation with **layer height** and **nozzle temperature**. Smaller layer heights and lower nozzle temperatures improved surface smoothness.
- Final models improved **error scores (MAE ↓ 25%, RMSE ↓ 30%)** and **prediction accuracy (R² ↑ 7%)** over baseline linear regression, demonstrating the effectiveness of advanced regression techniques.

## 🎯 Project Significance
This project provides a data-driven approach to optimizing 3D printing parameters, enabling:

✔️ **Improved print accuracy and strength** through precise parameter tuning  
✔️ **Reduced material waste** by identifying ideal settings before printing  
✔️ **Scalability** for different materials and printer models  

## 🚀 How to Install and Run the Project

### ✅ Prerequisites
Ensure you have the following installed:
- **Python 3**
- **Jupyter Notebook**

### 📥 Installation Steps
1️⃣ **Clone the Repository**
```bash
git clone https://github.com/ScottTorzewski/3D-Printing-ML-Model.git
```
2️⃣ **Navigate to the Project Directory**
```bash
cd 3D-Printing-ML-Model
```
3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```
4️⃣ **Launch Jupyter Notebook**
```bash
jupyter notebook
```
5️⃣ **Open the `.ipynb` file** and start running the cells!

## 🎯 How to Use the Project
1️⃣ **Run all notebook cells sequentially** to preprocess data, train models, and evaluate results.  
2️⃣ **Analyze performance metrics** for different regression models.  
3️⃣ **Adjust hyperparameters** to explore further optimizations.  
4️⃣ **Experiment with alternative models** like **Random Forest** or **SVR** for comparison.  

## 📂 Dataset
🔗 **Original Dataset:** [Kaggle - 3D Printer Data](https://www.kaggle.com/datasets/afumetto/3dprinter)

## 📜 License
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
