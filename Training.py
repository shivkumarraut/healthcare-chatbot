import re
import json
import pickle
import numpy as np
import random
from sklearn.neural_network import MLPClassifier

def simple_tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())

def lemmatize_simple(word):
    for suffix in ['ing', 'tion', 'ness', 'ment', 'ers', 'ed', 'es', 's']:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word

with open('data.json') as f:
    intents = json.load(f)

words, classes, documents = [], [], []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = simple_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatize_simple(w) for w in words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = [lemmatize_simple(w) for w in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    max_iter=1000,
    random_state=42
)
model.fit(np.array(train_x), np.array(train_y))
pickle.dump(model, open('model.pkl', 'wb'))
print(f"✅ Done! Accuracy: {model.score(np.array(train_x), np.array(train_y)):.4f}")
