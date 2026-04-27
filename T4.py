# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("twitter_training.csv", encoding='latin-1')

# Rename columns if needed
df.columns = ["ID", "Topic", "Sentiment", "Text"]

# ==============================
# 3. Data Cleaning
# ==============================
# Remove missing values
df.dropna(subset=["Text"], inplace=True)

# Convert text to string
df["Text"] = df["Text"].astype(str)

# ==============================
# 4. Sentiment Analysis (if not labeled)
# ==============================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# If dataset already has sentiment, skip this
df["Predicted_Sentiment"] = df["Text"].apply(get_sentiment)

# ==============================
# 5. Sentiment Distribution
# ==============================
plt.figure(figsize=(6,4))
sns.countplot(x="Predicted_Sentiment", data=df)
plt.title("Sentiment Distribution")
plt.show()

# ==============================
# 6. Sentiment by Topic/Brand
# ==============================
plt.figure(figsize=(10,5))
sns.countplot(x="Topic", hue="Predicted_Sentiment", data=df)
plt.xticks(rotation=90)
plt.title("Sentiment by Topic")
plt.show()

# ==============================
# 7. WordCloud (Positive)
# ==============================
positive_text = " ".join(df[df["Predicted_Sentiment"]=="Positive"]["Text"])

wordcloud = WordCloud(width=800, height=400).generate(positive_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Positive Sentiment WordCloud")
plt.show()

# ==============================
# 8. WordCloud (Negative)
# ==============================
negative_text = " ".join(df[df["Predicted_Sentiment"]=="Negative"]["Text"])

wordcloud = WordCloud(width=800, height=400).generate(negative_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Negative Sentiment WordCloud")
plt.show()
