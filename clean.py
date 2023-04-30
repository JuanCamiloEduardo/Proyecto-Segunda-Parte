import re
import nltk
import string 
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import cess_esp
nltk.download('cess_esp')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

def join_strings(string_list):
    """Toma una lista de strings y los une en un solo string separados por un espacio"""
    return ' '.join(string_list)

def clean_text(text):
    """
    Limpia el texto eliminando signos de puntuación, números y palabras vacías.
    """
    """ 
    newtext=[]
    for i in range(len(text)):
        newtext.append(text[i].lower())
        newtext[i] = re.sub('\[.*?\]', '', newtext[i])
        newtext[i] = re.sub('[%s]' % re.escape(string.punctuation), '', newtext[i])
        newtext[i] = re.sub('\w*\d\w*', '', newtext[i])
        newtext[i] = re.sub('[‘’“”…]', '', newtext[i])
        newtext[i] = re.sub('\n', '', newtext[i])
    """   
     # Convertir todo el texto a minúsculas   
     
    text= text.lower()
    text = re.sub('\[.*?\]', '', text) # Eliminar cualquier cosa entre corchetes
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Eliminar signos de puntuación
    text = re.sub('\w*\d\w*', '', text) # Eliminar cualquier número
    text = re.sub('[‘’“”…]', '', text) # Eliminar comillas
    text = re.sub('\n', '', text) # Eliminar saltos de línea
    print("Esta limpiando con CLean")
    return text

def remove_stopwords(text):
    """
    Elimina las palabras vacías del texto.
    """
    stop_words = set(stopwords.words('spanish'))
    
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def tokenize_text(text):
    """
    Tokeniza el texto.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def snowball_lemmatize(text):
    stemmer = SnowballStemmer(language='english')
    tokens = word_tokenize(text)
    lemmatized_text = [stemmer.stem(token) for token in tokens]
    return ' '.join(lemmatized_text)


