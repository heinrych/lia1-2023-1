
import nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import unicodedata
import unidecode
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # Baixar o recurso do WordNet
nltk.download('rslp')
nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from unidecode import unidecode
import os
os.environ["NUMBA_CUDA_DRIVER"] = "1"


class preprocess:
    
    def fix_encoding(text):
        vogais = ["a", "e", "i", "o", "u"]
        cedilha = "c"

        text = text.replace("√°", vogais[0]).replace("√†", vogais[0]).replace("√¢", vogais[0]).replace("√Å", vogais[0]).replace("√É", vogais[0]).replace("√£", vogais[0])
        text = text.replace("√©", vogais[1]).replace("√™", vogais[1]).replace("√â", vogais[1])
        text = text.replace("√ß", cedilha).replace("√á", cedilha)
        text = text.replace("√≠", vogais[2]).replace("√ç", vogais[2])
        text = text.replace("√≥", vogais[3]).replace("√µ", vogais[3]).replace("√¥", vogais[3]).replace("√ì", vogais[3])
        text = text.replace("√∫", vogais[4]).replace("√ö", vogais[4])
        
        text = text.replace("(c)", cedilha)

        entrada_normalizada = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
        return entrada_normalizada
    
    def expand_abbreviations(text):
        abbreviation_map = {
            'vc': 'voce',
            'tb': 'tambem',
            'pq': 'porque',
            'tbm': 'tambem',
            'q': 'que',
            'n': 'nao',
            'td': 'tudo',
            'pnc': 'pau no cu',
            'ta': 'esta',
            'blz': 'beleza',
            'vlw': 'valeu',
            'bjs': 'beijos',
            'flw': 'falou',
            'eh': 'e',
            'vdd': 'verdade',
            'dps': 'depois',
            'msm': 'mesmo',
            'qq': 'qualquer',
            'pf': 'por favor',
            't+' : 'ate mais',
            'sdd': 'saudade',
            'fzd': 'fazendo',
            'c': 'com',
            'd': 'de',
            'p': 'para',
            'vc': 'voce',
            'vc': 'voces',
            'tbm': 'tambem',
            'mt': 'muito',
            'pq': 'por que',
            'hj': 'hoje',
            'q': 'que',
            'cmg': 'comigo',
            'fds': 'fim de semana',
            'qnd': 'quando',
            'pqp':"puta que pariu",
            'n':'nao',
            'vsf':'vai se fuder',
            'vsfd':'vai se fuder',
            'sifoda':'se foda',
            'gnt':'gente',
            'vou t matar':'vou te matar',
            'gstz':'gostosa',
            'tdas':'todas',
            'tou':'estou',
            'to':'estou',
            'ta':'esta',
            'vox':'voce',
            'putaa':'puta',
            'muie':'mulher',
            'fdp':'filha da puta', #v8
            'calaboca':'cala boca' #v8
        }
        
        words = text.split()
        expanded_words = []
        
        for word in words:
            if word.lower() in abbreviation_map:
                expanded_words.append(abbreviation_map[word.lower()])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def preprocess_text(df):
        # Define a lista de stopwords em português
        stopwords_portuguese = stopwords.words('portuguese')
        stopwords_portuguese += ['http', 'https', 'www', 'com', 'br', 'rt']
        
        lemmatizer = WordNetLemmatizer()
        
        # Aplica as transformações em cada texto do dataframe
        df['texto_processado'] = df['review_text'].str.lower()  # Converte todo o texto para minúsculo # [pandas]
        df['texto_processado'] = df['texto_processado'].apply(preprocess.expand_abbreviations) # [pandas] substituição de termos por replace
        df['texto_processado'] = df['texto_processado'].apply(lambda text: preprocess.fix_encoding(text)) # [pandas]
        df['texto_processado'] = df['texto_processado'].apply(lambda text: re.sub(r'(.)\1{2,}', r'\1\1', text))  # Remove caracteres sequenciais repetidos
        df['texto_processado'] = df['texto_processado'].apply(lambda text: word_tokenize(text))  # Tokeniza o texto
        df['texto_processado'] = df['texto_processado'].apply(lambda tokens: [word for word in tokens if not word.startswith('@') and not word.startswith('#')])  # Remove tokens iniciados com @ e #
        df['texto_processado'] = df['texto_processado'].apply(lambda text: [unidecode(word) for word in text])  # Remove acentos
        df['texto_processado'] = df['texto_processado'].apply(lambda text: [re.sub(r'[^\w\s]', '', token) for token in text])  # Remove caracteres especiais
        df['texto_processado'] = df['texto_processado'].apply(lambda tokens: [word for word in tokens if word.isalpha()]) 
        df['texto_processado'] = df['texto_processado'].apply(lambda text: [lemmatizer.lemmatize(word) for word in text])  # Realiza lematização
        df['frase'] = df['texto_processado'].apply(lambda text: ' '.join(text))  # Reconstroi o texto
        
        df = df.drop_duplicates(subset=['frase'])  # Remove duplicatas
        df = df.dropna(subset=['frase'])  # Remove linhas com valores nulos
        df = df.dropna(subset=['polarity'])
        
        return df
   
   
    def preprocess_text_stop(df):
        # Define a list of stopwords in Portuguese
        stopwords_portuguese = stopwords.words('portuguese')
        stopwords_portuguese += ['http', 'https', 'www', 'com', 'br', 'rt']

        lemmatizer = WordNetLemmatizer()

        # Make a copy of the dataframe to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Apply transformations to each text in the dataframe
        df_copy.loc[:, 'texto_processado'] = df_copy['review_text'].str.lower()  # Convert all text to lowercase
        df_copy.loc[:, 'texto_processado'] = df_copy['texto_processado'].apply(preprocess.expand_abbreviations)
        df_copy.loc[:, 'texto_processado'] = df_copy['texto_processado'].apply(preprocess.fix_encoding)
        df_copy.loc[:, 'texto_processado'] = df_copy['texto_processado'].apply(lambda text: re.sub(r'(.)\1{2,}', r'\1\1', text))  # Remove sequential repeated characters

        df_copy.loc[:, 'texto_processado'] = df_copy['texto_processado'].apply(lambda text: [unidecode(word) for word in word_tokenize(text)])  # Remove accents
        df_copy.loc[:, 'texto_processado'] = df_copy['texto_processado'].apply(lambda tokens: [re.sub(r'[^\w\s]', '', token) for token in tokens])  # Remove special characters
        df_copy.loc[:, 'texto_processado'] = df_copy['texto_processado'].apply(lambda tokens: [word for word in tokens if word.isalpha()]) 
        df_copy.loc[:, 'texto_processado'] = df_copy['texto_processado'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])  # Perform lemmatization
        df_copy.loc[:, 'texto_processado'] = df_copy['texto_processado'].apply(lambda tokens: [word for word in tokens if word not in stopwords_portuguese and not word.startswith('@') and not word.startswith('#')])  # Remove tokens that are not words or that are stopwords
        df_copy.loc[:, 'frase'] = df_copy['texto_processado'].apply(lambda tokens: ' '.join(tokens))  # Reconstruct the text

        df_copy = df_copy.drop_duplicates(subset=['frase'])  # Remove duplicates
        df_copy = df_copy.dropna(subset=['frase'])  # Remove lines with null values

        return df_copy