from model.model import load_and_prepare_data, train_model, predict_emails

# Load data
df = load_and_prepare_data("data/fraud_email.csv")

# Train model
model, vectorizer = train_model(df)

# Tricky test cases
tricky_emails = [
    "Dear Employee, please update your password at this secure link immediately.",
    "Lunch at 1 PM? Bring the quarterly report if you can.",
    "URGENT: Your account has been suspended. Verify your identity to avoid termination.",
    "Don't forget the birthday cake for Sarah tomorrow!",
    "Congratulations! You've been selected for a confidential investment opportunity.",
    "Need that invoice from last week. Can you resend?",
    "Win a brand new iPhone by completing this short survey!"
]

# Predict
predictions, confidences = predict_emails(model, vectorizer, tricky_emails)

# Show results
for i, email in enumerate(tricky_emails):
    label = "FRAUD" if predictions[i] == 1 else "NORMAL"
    confidence = confidences[i][predictions[i]]
    print(f"\nEmail {i+1}: {email}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}")