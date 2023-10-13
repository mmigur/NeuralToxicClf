import string
import torch

from pymorphy2 import MorphAnalyzer

from transformers import AutoTokenizer, AutoModel

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

STOP_WORDS = stopwords.words('russian')
PUNCTUATION = string.punctuation
DIGITS = string.digits
TOKENIZER = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
RUBERT_MODEL = AutoModel.from_pretrained("cointegrated/rubert-tiny")

def preprocessing(text: str):
    def remove_punctuation(text: str) -> str:
        without_digits = (
            ''.join([word for word in text if (word not in PUNCTUATION) and (word not in DIGITS)])
        )
        return ' '.join([word for word in without_digits.split(' ') if word not in STOP_WORDS])

    def lemmatize(text: str) -> str:
        pymorphy2_analyzer = MorphAnalyzer()
        return ' '.join([pymorphy2_analyzer.parse(word)[0].normal_form.strip() for word in text.split(' ')]).strip()

    def vectorize(text: str):
        t = TOKENIZER(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = RUBERT_MODEL(**{key: value.to(RUBERT_MODEL.device) for key, value in t.items()})
        embeddings = model_output.last_hidden_state[:, :, :]
        embeddings = torch.nn.functional.normalize(embeddings)
    
        return embeddings.cpu().mean(dim = 1).squeeze(0).numpy()
    
    return vectorize(lemmatize(remove_punctuation(text)))