import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model.model import preprocess_text

# -------------------- Load Model --------------------
clf = joblib.load("model/rf_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# -------------------- Load Enron Data --------------------
df = pd.read_csv("data/enron_emails.csv")

# Preprocess and Vectorize
df['clean_message'] = df['clean_message'].astype(str).apply(preprocess_text)
X = vectorizer.transform(df['clean_message'])

# Predict
predictions = clf.predict(X)
df['prediction'] = predictions

# Count
fraud_count = sum(predictions)
normal_count = len(predictions) - fraud_count

print(f"\n📊 Total Emails: {len(predictions)}")
print(f"✅ Normal Emails: {normal_count}")
print(f"⚠ Fraudulent Emails: {fraud_count}")

# Save fraud emails
fraud_emails = df[df['prediction'] == 1]
fraud_emails.to_csv("data/enron_detected_fraud_emails.csv", index=False)
print("📁 Detected fraud emails saved to data/enron_detected_fraud_emails.csv")

# Show some fraud emails
print("\n📃 Detected Fraudulent Emails (first 5):\n")
for i, row in fraud_emails.head(5).iterrows():
    print(f"- {row['clean_message'][:100]}...")  # First 100 chars

# -------------------- Visualizations --------------------

# Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(
    [normal_count, fraud_count],
    labels=['Normal', 'Fraudulent'],
    autopct='%1.1f%%',
    colors=['#66b3ff', '#ff6666'],
    startangle=140
)
plt.title("📊 Email Classification: Pie Chart")
plt.axis('equal')
plt.show()

# Bar Chart
plt.figure(figsize=(6, 4))
sns.barplot(x=['Normal', 'Fraudulent'], y=[normal_count, fraud_count], palette='pastel')
plt.title("📊 Email Classification: Bar Chart")
plt.ylabel("Number of Emails")
plt.show()

# Heatmap (if true labels exist)
if 'class' in df.columns:
    cm = confusion_matrix(df['class'], df['prediction'])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='coolwarm', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title("🔥 Confusion Matrix Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()