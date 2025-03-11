# 🖨️ Using Regression to Approximate 3D Printing Parameters  

## 📌 Project Overview  
This project aims to develop a **machine learning model** that predicts the **print quality, accuracy, and strength** of a **3D printer's output**. These characteristics are quantified as:  
✅ **Roughness (μm)** – Surface texture measurement  
✅ **Elongation (%)** – Material stretchability before breaking  
✅ **Tensile Strength (MPa)** – Resistance to breaking under tension  

### 🔍 Approach  
1️⃣ **Exploratory Data Analysis (EDA)** to understand trends and relationships  
2️⃣ **Initial Linear Regression models** for all three target variables  
3️⃣ **Model engineering & optimization** to enhance prediction accuracy  
4️⃣ **Challenges**: Data cleaning, feature selection, and preprocessing before modeling  

**🛠️ Data Sources:**  
- **3D Printer:** Ultimaker S5  
- **Material Testing:** Sincotec GMBH Tester  

---

## 🚀 How to Install and Run the Project  

### ✅ Prerequisites  
Ensure you have the following installed before running the project:  
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
5️⃣ **Open the `.ipynb` file** in Jupyter and start running the cells!  

---

## 🎯 How to Use the Project  

1️⃣ **Run all notebook cells sequentially** to perform data preprocessing, training, and evaluation.  
2️⃣ **Visualize results** through plots, tables, and model performance metrics.  
3️⃣ **Optimize the model** by tweaking hyperparameters in the notebook.  
4️⃣ **(Optional)** Experiment with additional regression models to compare performance.  

---

## 📂 Dataset  
🔗 **Original Dataset:** [Kaggle - 3D Printer Data](https://www.kaggle.com/datasets/afumetto/3dprinter)  

---

## 📜 License  
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.  



