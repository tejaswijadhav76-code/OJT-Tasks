# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("bank.csv", sep=None, engine='python')
df.columns = df.columns.str.strip()

print("✅ Shape:", df.shape)
print("✅ Columns:", df.columns.tolist())
print(df.head())

# ==============================
# 3. Remove ID / Irrelevant Columns
# ==============================
# Drop columns that contain unique identifiers (like TXN029)
for col in df.columns:
    if df[col].dtype == 'object':
        if df[col].nunique() == len(df):  # Unique values = ID column
            print(f"❌ Dropping ID column: {col}")
            df.drop(col, axis=1, inplace=True)

# ==============================
# 4. Handle Missing Values
# ==============================
df.dropna(inplace=True)

# ==============================
# 5. Encode Categorical Columns
# ==============================
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print("✅ Data after encoding:\n", df.head())

# ==============================
# 6. Split Features & Target
# ==============================
target_col = 'y' if 'y' in df.columns else df.columns[-1]
print(f"✅ Target column: '{target_col}'")

X = df.drop(target_col, axis=1)
y = df[target_col]

print("✅ X shape:", X.shape)
print("✅ y shape:", y.shape)

# ==============================
# 7. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 8. Train Model
# ==============================
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ==============================
# 9. Predictions
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 10. Evaluation
# ==============================
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# 11. Visualize Decision Tree
# ==============================
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=['No', 'Yes'],
    filled=True,
    fontsize=9
)

plt.title("Decision Tree - Bank Marketing", fontsize=16)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()

print("✅ Decision tree saved as decision_tree.png")
