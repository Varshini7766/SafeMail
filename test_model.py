import pandas as pd
import joblib
import matplotlib.pyplot as plt
from model.model import preprocess_text

# Load model and vectorizer
clf = joblib.load("model/rf_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Load your test data (replace with the correct path to your test file)
df = pd.read_csv("data/test_emails.csv")
df['text'] = df['text'].astype(str).apply(preprocess_text)

# Vectorize
X = vectorizer.transform(df['text'])

# Predict
predictions = clf.predict(X)

# Count
fraud_count = sum(predictions)
normal_count = len(predictions) - fraud_count

print(f"\n📊 Total Emails: {len(predictions)}")
print(f"✅ Normal Emails: {normal_count}")
print(f"⚠ Fraudulent Emails: {fraud_count}")

# Plot
labels = ['Normal', 'Fraudulent']
sizes = [normal_count, fraud_count]
colors = ['#66b3ff', '#ff6666']

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title("Email Classification Result")
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.show()