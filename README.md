# Credit-card-default-prediction-system
---

# 🚀 Steps to Run the Credit Default Prediction Project

### **1. Clone / Download Project**

* Download this repository as a ZIP or clone it using GitHub Desktop.
* Open the folder in your IDE (PyCharm, VS Code, Jupyter, etc.).

---

### **2. Setup Environment**

* Make sure you have **Python 3.9 or higher** installed.
* Install the required libraries from `requirements.txt` using your IDE’s package manager.
* Required libraries:

  * pandas
  * scikit-learn
  * seaborn
  * matplotlib
  * joblib

---

### **3. Check Dataset Path**

* Open the file `train.py`.
* Ensure the dataset is loaded correctly:

```python
df = pd.read_csv("data.csv")
```

(If the path is hardcoded differently, change it to `"data.csv"`).

---

### **4. Train the Model**

* Run the script `train.py`.
* What happens after running:

  * The dataset is cleaned (missing values handled, outliers removed).
  * Categorical variables are converted into numerical form.
  * EDA plots (graphs) are generated and saved as `.png` files.
  * A **Random Forest Classifier** is trained.
  * Three files are generated automatically:

    * `model.joblib` → the trained model
    * `metrics.json` → contains accuracy and classification report
    * `feature_names.json` → list of features used in training

---

### **5. View Results**

* Open `metrics.json` to check accuracy and performance.
* Look for EDA plots like:

  * `target_distribution.png`
  * `numerical_histograms.png`
  * `correlation_heatmap.png`

---

### **6. Run Predictions**

* Create a file named `predict.py` with the provided code.
* Run it to test predictions on new data.
* The output will display whether the input represents **Default** or **No Default**.

---

### **7. (Optional) Cross-Validation**

* To check the model’s stability, you can add cross-validation code inside `train.py`.
* This will give you average accuracy across multiple folds.

---

### **8. Publish on GitHub**

* Upload all important files:

  * `train.py`
  * `predict.py`
  * `data.csv` (or a sample)
  * `requirements.txt`
  * `metrics.json`
  * `feature_names.json`
  * All plots (`.png`)
  * `README.md` (this file)
