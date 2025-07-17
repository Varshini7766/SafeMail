🛡️ SafeMail – Real-Time Email Fraud Detection Tool

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey?logo=flask)
![Status](https://img.shields.io/badge/Status-Deployed-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

SafeMail is a powerful web-based tool that helps users detect fraudulent emails using a machine learning model trained on real-world corporate communication data, including the infamous **Enron email dataset**.  

🔗 **Live Site**: [safemail-rneh.onrender.com](https://safemail-rneh.onrender.com)

📌 Features

✅ Upload one or more emails (CSV or text)  
✅ Get real-time predictions: `Fraudulent` or `Normal`  
✅ View confidence scores for each prediction  
✅ Download the results in CSV  
✅ Beautiful analytics via pie charts  
✅ Hosted and ready to use on the web

🧠 Tech Stack

| Area        | Tech Used                      |
|-------------|-------------------------------|
| Frontend    | HTML, CSS, JavaScript         |
| Backend     | Python (Flask)                |
| ML Model    | Random Forest + TF-IDF (n-grams) |
| Deployment  | Render                        |
| Visualization | JavaScript (Pie Charts)     |
| Dataset     | Enron Email Dataset           |

🚀 Try It Out

1. Go to [safemail-rneh.onrender.com](https://safemail-rneh.onrender.com)
2. Upload a `.csv` file with a column named `text`, or paste an email manually
3. Click **Verify**
4. View predictions + confidence scores
5. Explore results and download as needed

📂 Project Structure

SafeMail/
│
├── static/                 # CSS and JS assets
├── templates/              # HTML (Jinja) templates
│   ├── index.html
│   └── dashboard.html
├── model/                  # Trained ML model (.pkl)
├── app.py                  # Flask server
├── requirements.txt        # Python dependencies
├── README.md               # You're here!

🧪 Sample Output

| Email Text                  | Prediction  | Confidence |
|----------------------------|-------------|------------|
| "Please transfer $5000..." | Fraudulent  | 0.91       |
| "Team meeting at 3PM"      | Normal      | 0.96       |

---

🛠 Local Setup

Want to run it locally? Easy:

```bash
# Clone this repository
git clone https://github.com/Varshini7766/SafeMail.git
cd SafeMail

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

Visit `http://localhost:5000` in your browser.

📊 Future Improvements

[ ] Improve responsiveness and mobile view
[ ] Integrate user login and history
[ ] Add PDF and email body extraction support
[ ] Train with larger and more recent datasets

---
<img width="1914" height="516" alt="Screenshot 2025-07-17 130913" src="https://github.com/user-attach<img width="870" height="838" alt="Screenshot 2025-07-17 130946" src="https://github.com/user-attachments/assets/abc6811e-31bd-4b33-930c-f17b2b84eade" />
ments/assets/867ea3b0-2a15-4a2e-b52f-84657f79501d" />

👩‍💻 Author

Made with 💙 by [Varshini](https://github.com/Varshini7766)
📫 Reach out: [varshini@example.com](mailto:varshini@example.com) (replace with yours)

🌐 Live Demo
👉 Try it now: [https://safemail-rneh.onrender.com](https://safemail-rneh.onrender.com)

“Catch the fraud before it catches you. Stay smart, stay safe.” 😉
