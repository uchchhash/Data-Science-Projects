# Applied Machine Learning for Real-World Prediction Tasks

This repository presents two supervised learning projects that apply data science and machine learning techniques to real-world problems. The projects demonstrate complete ML workflows including exploratory data analysis (EDA), feature engineering, model building, performance evaluation, and interpretation of results.

---

## 📊 Project Summaries

### 1. 🚲 Bike Sharing Demand Prediction (Regression)

**Objective**:  
Predict the number of bikes rented in a given hour based on environmental and seasonal features such as temperature, humidity, weather conditions, and time.

**Notebooks**:
- `Regression Data Analysis (Bike sharing prediction).ipynb`  
  → EDA, preprocessing, feature transformations  
- `Regression model (Bike sharing prediction).ipynb`  
  → Model training, evaluation, and insights

**Techniques Used**:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

**Evaluation Metrics**:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

**Dataset**:  
[UCI Machine Learning Repository – Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

---

### 2. 🚢 Titanic Survival Classification (Classification)

**Objective**:  
Predict survival outcomes for Titanic passengers based on features such as age, sex, ticket class, and family relations.

**Notebooks**:
- `Data Analysis.ipynb`  
  → Data cleaning, EDA, feature exploration  
- `Data Modeling.ipynb`  
  → Training classifiers, performance evaluation

**Techniques Used**:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

**Evaluation Metrics**:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score

**Dataset**:  
[Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)

---

## 🧠 Key Concepts Demonstrated

- Handling missing and categorical data
- Feature encoding, scaling, and transformation
- Correlation analysis and feature selection
- Supervised learning: regression and classification
- Model comparison and hyperparameter tuning
- Visualization for both data and model evaluation

---

## 🛠 Tech Stack

| Tool            | Purpose                              |
|-----------------|--------------------------------------|
| Python          | Programming Language                 |
| Jupyter Notebook| Development Environment              |
| Pandas, NumPy   | Data manipulation and analysis       |
| Matplotlib, Seaborn | Data visualization              |
| Scikit-learn    | Machine learning models and metrics  |

---

## 🗂 Repository Structure

```

ML-Prediction-Projects/
├── Regression Data Analysis (Bike sharing prediction).ipynb
├── Regression model (Bike sharing prediction).ipynb
├── Data Analysis.ipynb
├── Data Modeling.ipynb
├── README.md
└── requirements.txt

````

---

## ▶️ How to Run the Notebooks

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/ML-Prediction-Projects.git
cd ML-Prediction-Projects
````

2. **(Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch the Jupyter notebooks**

```bash
jupyter notebook
```

---

## ✅ Results

* Successfully built predictive models with interpretable results.
* Applied both regression and classification algorithms on cleaned, preprocessed datasets.
* Compared models and identified the best-performing ones using relevant evaluation metrics.

---

## 👨‍💻 Author

**Uchchhash Sarkar**
Organization : Quantum.AI, Bangladesh

* [LinkedIn](https://www.linkedin.com/in/uchchhash)
* [GitHub](https://github.com/uchchhash)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

```
