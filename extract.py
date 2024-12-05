import random
import numpy as np
import pickle
import nltk
from transformers import BertTokenizer, BertForMaskedLM
import torch
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = BertForMaskedLM.from_pretrained("neuralmind/bert-base-portuguese-cased")

def neutralize_gender(word):
    """
    Neutraliza a palavra removendo a terminação de gênero para tratar 
    palavras com variações de gênero como 'obrigado' e 'obrigada'.
    """
    # Verifica se a palavra termina com "a", "o", "as", "os" e corta a última letra
    if word.endswith(("a", "o", "as", "os")):
        return word[:-1]  
    return word

def clear_writing(writing):
    """
    Processa a entrada de texto para neutralizar gênero e remover artigos.
    """
    stopwords = {"a", "o", "as", "os"}
    
    words = nltk.word_tokenize(writing)
    
    cleaned_words = []
    
    for word in words:
        lower_word = word.lower()
        
        if lower_word in stopwords:
            continue
        
        neutral_word = neutralize_gender(lower_word)
        
        cleaned_words.append(neutral_word)
    
    input_text = " ".join(cleaned_words)
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    return cleaned_words

# retornar array de saco de palavras: 0 ou 1 para cada palavra no saco que existe na frase

def bag_of_words(writing, words):
    """
        Pega as sentenças que são limpas e cria um pacote de palavras que são usadas 
        para classes de previsão que são baseadas nos resultados que obtivemos treinando o modelo.
    """
    # tokenize the pattern
    sentence_words = clear_writing(writing)

    # cria uma matriz de N palavras
    bag = [0]*len(words)
    for setence in sentence_words:
        for i, word in enumerate(words):
            if word == setence:
                # atribui 1 no pacote de palavra se a palavra atual estiver na posição da setença
                bag[i] = 1

    return(np.array(bag))

def class_prediction(writing, model):
    """
      Faz a previsao do pacote de palavras, usamos como limite de erro 0.25 para evitarmos overfitting
      e classificamos esses resultados por força da probabilidade.
    """

    # filtra as previsões abaixo de um limite 0.25
    prevision = bag_of_words(writing, words)
    response_prediction = model.predict(np.array([prevision]))[0]
    results = [[index, response] for index, response in enumerate(response_prediction) if response > 0.25]    

    # verifica nas previsões se não há 1 na lista, se não há envia a resposta padrão (anything_else) 
    # ou se não corresponde a margem de erro
    if "1" not in str(prevision) or len(results) == 0 :
        results = [[0, response_prediction[0]]]

    # classifica por força de probabilidade
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(intents, intents_json):
    """
        pega a lista gerada e verifica o arquivo json e produz a maior parte das respostas com a maior probabilidade.
    """
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    for idx in list_of_intents:
        if idx['tag'] == tag:
            # caso as respostas sejam um array contendo mais de uma, 
            # usamos a função de random para pegar uma resposta randomica da nossa lista
            result = random.choice(idx['responses'])
            break

    return result
