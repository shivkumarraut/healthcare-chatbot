import re
import os
import pickle
import numpy as np
import json
import random

from keras.models import load_model
from flask import Flask, render_template, request
from flask_cors import CORS

# Simple tokenizer - no NLTK dependency needed
def simple_tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

def lemmatize_simple(word):
    for suffix in ['ing', 'tion', 'ness', 'ment', 'ers', 'ed', 'es', 's']:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word

# Load model and data
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

app = Flask(__name__, template_folder='Templates')
app.static_folder = 'static'
CORS(app)


def bow(sentence):
    sentence_words = [lemmatize_simple(w) for w in simple_tokenize(sentence)]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    p = bow(sentence)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(ints):
    if not ints:
        return random.choice([
            "I'm not sure I understand. Could you rephrase?",
            "Try asking about drugs, blood pressure, hospitals, or pharmacies.",
            "Type 'help' to see what I can assist with."
        ])
    tag = ints[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I didn't get that. Please try again."


def chatbot_response(msg):
    try:
        ints = predict_class(msg)
        return get_response(ints)
    except Exception:
        return "Sorry, something went wrong. Please try again."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg', '')
    if not user_text.strip():
        return "Please enter a message."
    return chatbot_response(user_text)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
