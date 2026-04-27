import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────
# LOAD DATA (file should be in same folder)
# ─────────────────────────────
file_name = "population.csv"

try:
    df = pd.read_csv(file_name)
    print("✅ Dataset Loaded Successfully!")
except FileNotFoundError:
    print("❌ File not found!")
    print("👉 Make sure 'population.csv' is in the same folder")
    exit()

# Show available columns
print("\n📌 Columns in dataset:", list(df.columns))

# ─────────────────────────────
# COLUMN NAMES
# ─────────────────────────────
age_col = "Age"
gender_col = "Gender"

if age_col not in df.columns or gender_col not in df.columns:
    print("❌ Column names not found!")
    print("👉 Update column names from above list")
    exit()

# ─────────────────────────────
# CLEAN DATA
# ─────────────────────────────
df = df.dropna(subset=[age_col, gender_col])

# ─────────────────────────────
# PLOT GRAPH
# ─────────────────────────────
plt.figure(figsize=(10, 5))

# Histogram (Age)
plt.subplot(1, 2, 1)
plt.hist(df[age_col], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

# Bar Chart (Gender)
plt.subplot(1, 2, 2)
df[gender_col].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
