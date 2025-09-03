import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Blue-based theme
sns.set(style="whitegrid", palette="Blues")

# ============================
# 1. Load Dataset
# ============================
df = pd.read_csv("D:\\credit\\data.csv")  # Change path if needed

print("=== Original Data Info ===")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDataset shape:", df.shape)

# ============================
# 2. Data Cleaning
# ============================
# Fill missing numeric values with median
for col in df.select_dtypes(include=["float64", "int64"]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove outliers (IQR method)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nData after cleaning:")
print(df.info())

# ============================
# 3. Encode categorical variables early
# ============================
categorical_cols = ["home_ownership", "purpose", "employment_length"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ============================
# 4. Data Visualizations
# ============================

# Target variable distribution (blue shades)
plt.figure(figsize=(5, 4))
sns.countplot(x="default", data=df, hue="default", palette="Blues", legend=False)
plt.title("Target Variable Distribution")
plt.xticks([0, 1], ["No Default", "Default"])
plt.savefig("target_distribution.png")
plt.show()

# Histograms for numerical features
df[numeric_cols].hist(
    bins=20, figsize=(12, 8), color='#1f77b4', edgecolor='black'
)
plt.suptitle("Numerical Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig("numerical_histograms.png")
plt.show()

# Boxplots (only top correlated features after encoding)
top_boxplot_features = ["loan_amnt", "annual_inc", "int_rate"]
for col in top_boxplot_features:
    if col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="default", y=col, data=df, palette="Blues")
        plt.title(f"{col} vs Default")
        plt.savefig(f"boxplot_{col}.png")
        plt.show()

# Countplots for categorical variables
for col in ["home_ownership", "purpose", "employment_length"]:
    plt.figure(figsize=(7, 5))
    sns.countplot(x=col, hue="default", data=df, palette="Blues")
    plt.title(f"{col} vs Default")
    plt.xticks(rotation=45)
    plt.savefig(f"countplot_{col}_vs_default.png")
    plt.show()

# Correlation heatmap (on encoded data only)
plt.figure(figsize=(10, 8))
corr = df_encoded.corr()
sns.heatmap(corr, annot=False, cmap="Blues")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Top correlated features with target
top_features = corr["default"].abs().sort_values(ascending=False)[1:6]
plt.figure(figsize=(6, 4))
top_features.plot(kind="bar", color="#1f77b4", edgecolor="black")
plt.title("Top 5 Correlated Features with Default")
plt.ylabel("Correlation (absolute)")
plt.savefig("top_correlated_features.png")
plt.show()

# ============================
# 5. Train-Test Split
# ============================
X = df_encoded.drop("default", axis=1)
y = df_encoded["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 6. Train Model
# ============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ============================
# 7. Evaluation
# ============================
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("\nModel Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc, "report": report}, f, indent=4)

joblib.dump(model, "model.joblib")
print("[done] Saved model.joblib, metrics.json")

with open("feature_names.json", "w") as f:
    json.dump(list(X.columns), f, indent=4)
