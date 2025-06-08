import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import skew

# stop all warning messages
warnings.filterwarnings("ignore")
output_dir = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(output_dir, exist_ok=True)

# load dataset
df = pd.read_csv('../data/credit_dataset.csv')

# drop useless columns ===
df.drop(columns=["ZipCode", "DriversLicense"], axis=1, inplace=True)

# transform skewed features
df["Debt"] = np.log1p(df["Debt"])
df["Age"] = np.sqrt(df["Age"])
df["CreditScore"] = np.log1p(df["CreditScore"])
df["Income"] = np.log1p(df["Income"])
df["YearsEmployed"] = np.log1p(df["YearsEmployed"])

# print skewness
print("\n========== SKEWNESS ==========")
print(f"Skewness pentru Age: {skew(df['Age'])}")
print(f"Skewness pentru Debt: {skew(df['Debt'])}")
print(f"Skewness pentru CreditScore: {skew(df['CreditScore'])}")

# binary column distributions
binary_cols = ["Gender", "Married", "BankCustomer", "PriorDefault", "Employed", "Approved"]
fig, axes = plt.subplots(1, len(binary_cols), figsize=(15, 4))
fig.suptitle("Distribuții pentru coloane binare", fontsize=14)
for i, col in enumerate(binary_cols):
    sns.countplot(x=df[col], ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# boxplots after transformations
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("Boxplot-uri după transformări logaritmice", fontsize=14)
sns.boxplot(x=df["Debt"], ax=axes[0])
axes[0].set_title("Debt")
sns.boxplot(x=df["Age"], ax=axes[1])
axes[1].set_title("Age")
sns.boxplot(x=df["CreditScore"], ax=axes[2])
axes[2].set_title("CreditScore")
sns.boxplot(x=df["Income"], ax=axes[3])
axes[3].set_title("Income")
sns.boxplot(x=df["YearsEmployed"], ax=axes[4])
axes[4].set_title("YearsEmployed")
plt.tight_layout()
plt.show()

# label encode categorical features
non_binary_cols = [c for c in df.columns if df[c].dtype == "object"]
label_encoders = {}
def ApplyEncoder(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder  # save the encoder
    return df
for col in non_binary_cols:
    df = ApplyEncoder(df, col)

# correlation heatmap
plt.figure(figsize=(12, 8))
fig.suptitle("Heatmap", fontsize=14)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=45)
plt.title("Heatmap")
plt.show()

# check similarity between Married and BankCustomer
print("\n========== CORELATIE MARRIED vs BANKCUSTOMER ==========")
correlation = df[["Married", "BankCustomer"]].corr()
print(correlation)
identical_percentage = (df["Married"] == df["BankCustomer"]).sum() / len(df)
print(f"Identical: {identical_percentage * 100:.2f}%")
# print(df["Married"].value_counts(normalize=True))
# print(df["BankCustomer"].value_counts(normalize=True))

# crain/test split
X = df.drop(columns=["Approved"])
y = df["Approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model with BankCustomer
model1 = RandomForestClassifier(random_state=1)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)

# model without BankCustomer
X_train = X_train.drop(columns=["BankCustomer"])
X_test = X_test.drop(columns=["BankCustomer"])
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)

print("\n========== RANDOM FOREST COMPARATIV ==========")
print(f"Accuracy with BankCustomer: {acc1:.4f}")
print(f"Accuracy wo BankCustomer: {acc2:.4f}")


# evaluation function
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1-Score: {f1:.4f}")
    # print("\nClasifiction:\n", classification_report(y_true, y_pred))
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}

# evaluate both models
metrics = evaluate_model(y_test, y_pred1)
metrics2 = evaluate_model(y_test, y_pred2)

# train and compare multiple model
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(min_data_in_leaf=5, min_gain_to_split=0.0, verbose=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = evaluate_model(y_test, y_pred)

# model comparison
print("\n========== SCORURI COMPARATIVE ==========")
comparison = pd.DataFrame(results).T
print(comparison)

trained_models = {}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    trained_models[name] = model  # keep the trained model
    results[name] = evaluate_model(y_test, y_pred)

# choose the best model from those already trained
best_model_name = max(results, key=lambda name: results[name]["F1-Score"])
best_model = trained_models[best_model_name]

print(f"\n Best model: {best_model_name}")

# rebuild X_train completely to include all columns in the correct order
X_full = df.drop(columns=["Approved"])
y_full = df["Approved"]

# final data split
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# train the best model, assuming it's RandomForest
final_model = RandomForestClassifier()
final_model.fit(X_train_full, y_train_full)

# save
joblib.dump(final_model, os.path.join(output_dir, "credit_approval_model.pkl"))
joblib.dump(label_encoders, os.path.join(output_dir, "label_encoders.pkl"))



