# ============================================
# US ACCIDENT ANALYSIS (IDLE FRIENDLY VERSION)
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset...")

# Load only required columns (FASTER)
cols = ['Start_Time', 'Start_Lat', 'Start_Lng',
        'Weather_Condition', 'Severity',
        'Traffic_Signal', 'Junction', 'Roundabout']

df = pd.read_csv("US_Accidents_March23.csv", usecols=cols)

print("Original Shape:", df.shape)

# Take SAMPLE to avoid lag
df = df.sample(50000)

print("Sample Loaded:", df.shape)

# ============================================
# Data Processing
# ============================================

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour
df['Day'] = df['Start_Time'].dt.day_name()

df.dropna(inplace=True)

print("Data Cleaned")

# ============================================
# Plot 1: Hour Distribution
# ============================================

print("Plotting Hour Graph...")

plt.figure()
sns.countplot(x='Hour', data=df)
plt.title("Accidents by Hour")
plt.show()

# ============================================
# Plot 2: Day Distribution
# ============================================

print("Plotting Day Graph...")

plt.figure()
sns.countplot(x='Day', data=df,
              order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.xticks(rotation=45)
plt.title("Accidents by Day")
plt.show()

# ============================================
# Plot 3: Weather
# ============================================

print("Plotting Weather Graph...")

top_weather = df['Weather_Condition'].value_counts().head(10)

plt.figure()
sns.barplot(x=top_weather.values, y=top_weather.index)
plt.title("Top Weather Conditions")
plt.show()

# ============================================
# Plot 4: Hotspots
# ============================================

print("Plotting Hotspots...")

plt.figure()
sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df, alpha=0.3)
plt.title("Accident Hotspots")
plt.show()

# ============================================
# Final Output
# ============================================

print("\n===== FINAL INSIGHTS =====")
print("Peak Hours:\n", df['Hour'].value_counts().head())
print("Top Weather:\n", df['Weather_Condition'].value_counts().head())
print("Severity:\n", df['Severity'].value_counts())
