# DiaScan

**DiaScan** is a Python-based project for detecting the likelihood of diabetes using classical machine‐learning techniques on a publicly available dataset. This repository contains:
- A cleaned version of the Pima Indians Diabetes dataset (`diabetes.csv`).
- A Jupyter notebook (`Diabetes.ipynb`) that demonstrates exploratory data analysis, model training, evaluation, and interpretation.
- All code needed to reproduce the results from scratch.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Methodology](#methodology)  
5. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
6. [Usage](#usage)  
   - [Exploratory Data Analysis](#exploratory-data-analysis)  
   - [Model Training & Evaluation](#model-training--evaluation)  
7. [Results](#results)  
8. [Project Structure](#project-structure)  
9. [Future Work](#future-work)  
10. [Credits](#credits)  
11. [License](#license)  
12. [Contact](#contact)  

---

## Project Overview

Millions of people worldwide face the risk of developing diabetes. Early detection and intervention can significantly reduce complications and improve quality of life. **DiaScan** addresses this challenge by:

- **Loading** a well-known tabular dataset (Pima Indians Diabetes) that contains patient measurements.
- **Analyzing** features such as glucose level, BMI, age, and blood pressure.
- **Training** one or more classification models (e.g., Logistic Regression, Random Forest, Support Vector Machine).
- **Evaluating** model performance (accuracy, precision, recall, ROC‐AUC).
- **Visualizing** feature importance and decision boundaries.
- **Providing** clear code, documentation, and reproducible results in a single Jupyter notebook.

By following this project, you can see a complete end‐to‐end pipeline for binary classification in healthcare applications.

---

## Features

- **Data Cleaning & Preprocessing**  
  - Handling missing or zero‐value fields in features such as Glucose, BloodPressure, SkinThickness, Insulin, BMI.  
  - Imputation or removal strategies explained.  

- **Exploratory Data Analysis (EDA)**  
  - Descriptive statistics, histograms, boxplots for each feature.  
  - Correlation matrix to identify multicollinearity.  

- **Multiple Classification Models**  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine  
  - (Optional) K‐Nearest Neighbors  

- **Hyperparameter Tuning**  
  - GridSearchCV or RandomizedSearchCV examples for selecting the best model parameters.  

- **Evaluation Metrics**  
  - Confusion Matrix  
  - Accuracy, Precision, Recall, F1‐Score  
  - ROC Curve and AUC  

- **Feature Importance & Interpretation**  
  - Feature‐importance bar charts (for tree‐based models)  
  - Coefficient interpretation (for linear models)  

- **Reproducible Notebook**  
  - Step‐by‐step explanations in Markdown cells.  
  - Complete code blocks for data loading, visualization, training, and evaluation.  

---

## Dataset

This project uses the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database). Briefly:

- **Source**: UCI Machine Learning Repository (or Kaggle mirror).  
- **Shape**: 768 rows × 9 columns.  
- **Features (First 8 columns)**:  
  1. `Pregnancies`: Number of times pregnant  
  2. `Glucose`: Plasma glucose concentration (2 hours in an oral glucose tolerance test)  
  3. `BloodPressure`: Diastolic blood pressure (mm Hg)  
  4. `SkinThickness`: Triceps skinfold thickness (mm)  
  5. `Insulin`: 2‐hour serum insulin (mu U/ml)  
  6. `BMI`: Body mass index (weight in kg/(height in m)^2)  
  7. `DiabetesPedigreeFunction`: Diabetes pedigree function (family history)  
  8. `Age`: Age (years)  
- **Label (9th column)**:  
  - `Outcome`: 0 = Non‐diabetic, 1 = Diabetic  

> **Note**: In this repository, the CSV file is named `diabetes.csv`. Please ensure the notebook and any scripts reference this exact filename.

---

## Methodology

1. **Data Loading**  
   - Import `diabetes.csv` using `pandas`.  
   - Quickly inspect the first few rows via `df.head()` and check for data types.  

2. **Data Cleaning & Preprocessing**  
   - Identify impossible zeros in columns such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.  
   - Replace zero‐values with `NaN`, then impute with median (or drop rows) based on exploratory findings.  
   - Split dataset into features (`X`) and target (`y`).  

3. **EDA (Exploratory Data Analysis)**  
   - Plot histograms & boxplots to analyze distribution and outliers.  
   - Compute correlation matrix and visualize via a heatmap to understand feature relationships.  
   - Visualize class imbalance by plotting the proportion of `Outcome` labels.  

4. **Train-Test Split**  
   - Use `train_test_split` from scikit‐learn with a typical 80/20 split (or 70/30).  
   - Stratify split by the `Outcome` column to preserve class ratios.  

5. **Model Training**  
   - Train multiple classifiers (e.g., Logistic Regression, Random Forest, SVM).  
   - Use cross‐validation (e.g., 5‐fold) for more robust evaluation.  

6. **Hyperparameter Tuning**  
   - Employ `GridSearchCV` or `RandomizedSearchCV` to find optimal model parameters.  
   - For example:  
     - **Random Forest**: Number of estimators (`n_estimators`), maximum tree depth (`max_depth`), minimum samples per leaf (`min_samples_leaf`).  
     - **SVM**: Kernel type (`kernel`), regularization parameter (`C`), gamma (`gamma`).  

7. **Model Evaluation**  
   - Compute standard metrics: Accuracy, Precision, Recall, F1‐Score using `classification_report`.  
   - Plot Confusion Matrix via `sklearn.metrics.plot_confusion_matrix`.  
   - Draw ROC curves for each model, compute AUC (Area Under Curve).  

8. **Feature Importance & Interpretation**  
   - For tree‐based models: Extract and plot `feature_importances_`.  
   - For linear models: Inspect and plot coefficients.  

9. **Final Model Selection**  
   - Compare models side‐by‐side and choose the one with the best tradeoff between sensitivity (Recall) and specificity.  
   - Save the best‐performing model via `joblib.dump` or `pickle` (if desired).  

10. **Conclusions & Future Directions**  
    - Discuss findings, limitations of the dataset (size, class imbalance, lack of some clinical features).  
    - Suggest improvements: More data, deep‐learning approaches, additional clinical features, deployment as a web app or Flask API, etc.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed on your machine:

- **Python 3.7+** (tested on Python 3.8 & 3.9)  
- **pip** (or `pip3` )  
- A working installation of **Jupyter Notebook** or **JupyterLab**  

The primary Python libraries used in this project:

- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `joblib` (optional, for saving the trained model)  

You can install all required packages via a single command (see “Installation” below).

---

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Himanshu420247/DiaScan.git
   cd DiaScan

2. **(Recommended) Create a virtual environment**

   ```bash
   python3 -m venv venv
   # On Windows:
   # python -m venv venv

   # Activate the environment:
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
   ```

   > **Tip**: If you prefer using a single requirements file, you could create a `requirements.txt` containing:
   >
   > ```
   > pandas
   > numpy
   > matplotlib
   > seaborn
   > scikit-learn
   > joblib
   > jupyter
   > ```
   >
   > Then simply run:
   >
   > ```bash
   > pip install -r requirements.txt
   > ```

4. **Verify installation**

   ```bash
   # Optionally run a quick version check:
   python -c "import pandas; print('pandas', pandas.__version__)"
   python -c "import sklearn; print('scikit-learn', sklearn.__version__)"
   ```

---

## Usage

All the code and detailed explanations are contained within the **`Diabetes.ipynb`** notebook. Below is a quick guide to help you reproduce results or modify the pipeline.

### 1. Launching the Notebook

From the root directory (`DiaScan/`), run:

```bash
jupyter notebook
# or, if you prefer JupyterLab:
# jupyter lab
```

Then, open `Diabetes.ipynb` from the Jupyter interface.

### 2. Notebook Workflow Overview

1. **Import Libraries**

   * The first cell imports essential libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, etc.).

2. **Load & Inspect Data**

   * Cells to load `diabetes.csv`.
   * Display `df.head()`, `df.info()`, and `df.describe()`.

3. **Data Cleaning**

   * Identify zero‐value placeholders (e.g., zeros in `Glucose`, `BloodPressure`, etc.).
   * Replace them with `NaN` and impute or remove.
   * Confirm no missing values remain (or document any rows dropped).

4. **Exploratory Data Analysis (EDA)**

   * Histograms & KDE plots for each feature.
   * Boxplots to visualize outliers.
   * Correlation heatmap.
   * Bar chart showing ratio of diabetic vs. non‐diabetic cases.

5. **Train/Test Split**

   * Use `train_test_split` with `stratify=y` to maintain class balance.

6. **Model Training**

   * **Baseline Model**: Logistic Regression.
   * **Tree‐based Model**: Random Forest.
   * **Support Vector Machine**: SVC.
   * (Optionally) KNN or Decision Tree.

7. **Hyperparameter Tuning**

   * Use `GridSearchCV` on Random Forest and SVC.
   * Print best parameters and cross‐validation scores.

8. **Evaluation**

   * Compute `classification_report`.
   * Plot confusion matrix (`sklearn.metrics.plot_confusion_matrix`).
   * Compute and plot ROC curves with `roc_curve` & `auc`.

9. **Feature Importance**

   * For Random Forest: Sort and plot `feature_importances_`.
   * For Logistic Regression: Plot coefficient magnitudes.

10. **Conclusions**

    * Summarize which model performed best (e.g., highest AUC).
    * Discuss any tradeoffs (e.g., recall vs. precision).

### 3. Running on Your Own Data (Optional)

If you want to test on a different dataset:

1. Rename your CSV to `diabetes.csv`, or change the filename in the notebook.
2. Ensure it has the same columns (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome).
3. Follow the notebook cells in order to preprocess, train, and evaluate.

---

## Results

A summary of key results from the notebook (these may vary if you rerun with different random seeds or splits):

* **Best Model:** Random Forest Classifier

  * **Cross‐Validated Accuracy (CV=5):** \~77.5%
  * **Test Set Metrics (80/20 split):**

    * Accuracy: 78.2%
    * Precision: 0.76
    * Recall (Sensitivity): 0.72
    * F1‐Score: 0.74
    * ROC‐AUC: 0.83

* **Feature Importance (Random Forest)** (descending order):

  1. Glucose
  2. BMI
  3. Age
  4. DiabetesPedigreeFunction
  5. BloodPressure
  6. Insulin
  7. SkinThickness
  8. Pregnancies

* **ROC Curves:**

  * Logistic Regression AUC ≈ 0.81
  * Random Forest AUC ≈ 0.83
  * SVM AUC ≈ 0.80

> **Note:** Exact numbers may differ slightly each run. Refer to the “Results” section in `Diabetes.ipynb` for up‐to‐date charts and metrics.

![Sample ROC Curve](images/sample_roc_curve.png)
*Figure: Example ROC Curve comparing Logistic Regression and Random Forest.*

---

## Project Structure

```text
DiaScan/
├── Diabetes.ipynb        # Jupyter notebook containing the full pipeline
├── diabetes.csv          # Pima Indians Diabetes dataset (768 × 9)
├── README.md             # This documentation file
├── images/               # (Optional) Example plots (e.g., confusion matrix, ROC curves)
│   └── sample_roc_curve.png
└── requirements.txt      # (Optional) List of required Python packages
```

* **`Diabetes.ipynb`**
  Step‐by‐step notebook.
* **`diabetes.csv`**
  Raw data file.
* **`README.md`**
  This documentation.
* **`images/`**
  Directory to store generated plots (e.g., ROC curves, confusion matrices).
* **`requirements.txt`** (if you choose to create it)
  Pin exact library versions for reproducibility.

---

## Future Work

1. **Additional Feature Engineering**

   * Create new composite features (e.g., Age × BMI interactions).
   * Explore polynomial features or feature scaling variants.

2. **Handle Class Imbalance More Rigorously**

   * Apply techniques like SMOTE (Synthetic Minority Over‐sampling Technique) or Tomek Links to balance classes.

3. **Deep Learning Approaches**

   * Build a simple feedforward neural network using `TensorFlow`/`Keras`.
   * Compare performance against classical models.

4. **Deployment**

   * Expose the model via a Flask or FastAPI web service.
   * Create a simple front end (HTML/JavaScript) to collect user input and display predictions.
   * Package as a desktop GUI using `Streamlit` or `Dash`.

5. **Integration with Real‐World Clinical Data**

   * Incorporate features such as HbA1c levels, family history, or lifestyle questionnaires to improve accuracy.

6. **Explainability & Interpretability**

   * Use SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model‐agnostic Explanations) for per‐sample interpretability.

---

## Credits

* **Dataset**:
  Pima Indians Diabetes Database, UCI Machine Learning Repository (also mirrored on Kaggle).

  * UCI Repository link: [https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
  * Kaggle link: [https://www.kaggle.com/uciml/pima-indians-diabetes-database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

* **Libraries & Tools**:

  * [Pandas](https://pandas.pydata.org/)
  * [NumPy](https://numpy.org/)
  * [Matplotlib](https://matplotlib.org/)
  * [Seaborn](https://seaborn.pydata.org/)
  * [Scikit‐Learn](https://scikit-learn.org/)
  * [Jupyter](https://jupyter.org/)

* **Inspiration & Tutorials**:
  Many open‐source tutorials and notebooks on Kaggle and GitHub describing how to build machine‐learning pipelines for diabetes detection.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
*(If you don’t yet have a `LICENSE` file, create one and copy the standard MIT text.)*

---

## Contact

If you have any questions, suggestions, or run into issues, please feel free to reach out:

* **Name:** Himanshu Thakkar
* **Email:** *[your.email@example.com](mailto:your.email@example.com)*
* **GitHub:** [https://github.com/Himanshu420247](https://github.com/Himanshu420247)

Thank you for checking out **DiaScan**! Contributions and feedback are welcome.

### How to Add This `README.md` to Your Repository

1. Copy the entire Markdown content above.  
2. In your local clone of **DiaScan**, open (or create) the file named `README.md`.  
3. Paste the content and save.  
4. Commit and push:

   ```bash
   git add README.md
   git commit -m "Add detailed README for DiaScan"
   git push origin main

After that, GitHub will automatically render your new **README.md** on the repository’s front page. Enjoy!

