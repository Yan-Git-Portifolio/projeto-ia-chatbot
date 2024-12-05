import json
import pickle
import nltk
import random
import numpy as np

#removendo um warning chato
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Inicializamos nossa lista de palavras, classes, documentos e definimos quais palavras serão ignoradas
words = []
documents = []
# é feita a leitura do arquivo intents.json e transformado em json
intents = json.loads(open('intents.json', encoding='utf-8').read())
# adicionamos as tags em nossa lista de classes
classes = [i['tag'] for i in intents['intents']]
ignore_words = ["!", "@", "#", "$", "%", "*", "?"]

# percorremos nosso array de objetos
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizamos os patterns e adicionamos na lista de palavras
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # adiciona aos documentos para identificarmos a tag para a mesma
        documents.append((word, intent['tag']))

# Lematizamos as palavras ignorando as da lista ignore_words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Classificamos nossas listas
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Salvamos as palavras e classes nos arquivos pkl
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inicializamos o treinamento
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Criamos nosso conjunto de palavras com 1 se a correspondência de palavras for encontrada no padrão atual
    bag = [1 if word in pattern_words else 0 for word in words]

    # Verificamos a consistência do comprimento da bag
    if len(bag) != len(words):
        print(f"Tamanho da bag inconsistente: {len(bag)}")

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Verificamos a consistência do comprimento dos elementos em training
for i, (bag, output) in enumerate(training):
    if len(bag) != len(words) or len(output) != len(classes):
        print(f"Erro na posição {i}: bag tem tamanho {len(bag)}, output tem tamanho {len(output)}")

# Embaralhamos nosso conjunto de treinamentos e transformamos em numpy array
random.shuffle(training)
x = np.array([item[0] for item in training], dtype=np.float32)
y = np.array([item[1] for item in training], dtype=np.float32)

# Criamos nosso modelo com 3 camadas
model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Compilamos o modelo
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Ajustamos e salvamos o modelo
model.fit(x, y, epochs=200, batch_size=5, verbose=1)
model.save('model.keras')

print("Fim")

