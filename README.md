# 🏥 MediBot — Healthcare Assistant Chatbot

A domain-specific healthcare chatbot built with Python, Flask, and a custom-trained neural network using Keras and NLP. MediBot helps users find information on drug reactions, blood pressure tracking, hospitals, pharmacies, and general health tips.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-2.13.1-red?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-orange?style=flat-square)

---

## 🚀 Live Demo

[View Live on Render](#) *(link after deployment)*

---

## ✨ Features

- 🤖 Custom-trained neural network for intent classification (3-layer Keras model)
- 🧠 Full NLP pipeline: tokenisation → lemmatisation → bag-of-words → classification
- 💊 Adverse drug reaction information
- 🩺 Blood pressure tracking and search
- 🏥 Hospital search by name, location, and type
- 🏪 Pharmacy search
- 💡 General health tips and wellness advice
- 🚨 Emergency guidance with helpline numbers
- ⚡ Real-time responses via Flask REST API
- 📱 Responsive, modern chat UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML Model | Keras (TensorFlow), Neural Network |
| NLP | NLTK (tokenisation, lemmatisation) |
| Frontend | HTML, CSS, Vanilla JS |
| Deployment | Render |

---

## 📁 Project Structure

```
healthcare-chatbot/
├── app.py              # Flask app & chatbot logic
├── Training.py         # Model training script
├── data.json           # Intents & training data
├── model.h5            # Trained Keras model
├── texts.pkl           # Vocabulary (pickle)
├── labels.pkl          # Intent classes (pickle)
├── requirements.txt    # Dependencies
├── Procfile            # Render deployment config
├── Templates/
│   └── index.html      # Chat UI
└── static/
    └── styles/
        └── style.css   # Styles
```

---

## ⚙️ How It Works

1. User sends a message via the chat UI
2. Flask receives the message at `/get` endpoint
3. The NLP pipeline processes the text:
   - Tokenises the sentence using NLTK
   - Lemmatises each word to its base form
   - Converts to a bag-of-words vector
4. The trained Keras model predicts the intent
5. A random response from the matched intent is returned
6. The UI displays the response with a typing animation

---

## 🏃 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/shivkumarraut/healthcare-chatbot.git
cd healthcare-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

### 5. Retrain the model (optional)
If you update `data.json` with new intents:
```bash
python Training.py
```

---

## 🌐 Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python version**: 3.9
5. Click Deploy 🚀

---

## 💬 Sample Conversations

| You say | MediBot responds |
|---------|-----------------|
| "Drug side effects" | Opens adverse drug reaction module |
| "Find a hospital near me" | Asks for location/hospital name |
| "Give me a health tip" | Shares daily wellness tips |
| "Medical emergency" | Provides emergency helpline numbers |
| "Track my blood pressure" | Opens BP tracking module |

---

## ⚠️ Disclaimer

MediBot provides **general health information only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 👨‍💻 Author

**Shiv Kumar Raut**
- GitHub: [@shivkumarraut](https://github.com/shivkumarraut)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
