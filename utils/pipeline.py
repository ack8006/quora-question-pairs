import re
from string import punctuation

from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer

# import nltk
# nltk.download('stopwords')
# nltk.download('snowball_data')
# nltk.download('wordnet')


#Must return list of words
def pipeline(text, rm_stop_words=0, rm_punc=False, number_token=False, 
                keep_questions=False, stemmer=None, lemmatizer=None):
    '''
    Arguments:
    text: STRING
    rm_stop_words: INT: 1-Remove, 0-Nothing, -1-token
    rm_punc: BOOL True - remove all punctuation
    number_token: BOOL: True replace numbers with token, False-Nothing
    keep_questions: BOOL: True-do not remove question words 5Ws, how, which, can
    stem: stemmer object: SnowballStemmer('english') object
    lemma: BOOL: True- WordNetLemmatizer() object
        CANNOT BE STEM AND LEMMA
    
    Return: LIST: parsed words
    '''

    assert rm_stop_words in (-1, 0, 1)
    assert rm_punc in (-1, 0, 1)
    assert not (stemmer and lemmatizer)

    text = text.lower()

    text = text.replace("who's", "who is")
    text = text.replace("what's", "what is")
    text = text.replace("when's", "when is")
    text = text.replace("where's", "where is")
    text = text.replace("why's", "why is")
    text = text.replace("how's", "how is")

    text = text.replace("'s", "")
    text = text.replace("'ve", " have")
    text = text.replace("can't", "cannot")
    text = text.replace("n't", " not")
    text = text.replace("i'm", "i am")
    text = text.replace("'re", " are")
    text = text.replace("'d", " would")
    text = text.replace("'ll", " will")
    text = text.replace(',', '')
    #Replace k thousand with actual number
    text = re.sub(r"([0-9]+)(k)", r"\1,000", text)

    if number_token:
        text = re.sub(r'[,.0-9]+', " NUMBERTOKEN ", text)

    #spaces around non number punctuation
    text = re.sub(r'([a-zA-z]+)([^\w\s])([a-zA-z]*)', r"\1 \2 \3", text)

    #Removes excess whitespace
    text = re.sub(r"\s{2,}", " ", text)

    if rm_punc:
        text = ''.join([x if x not in punctuation else '' for x in text])
        # text = text.translate(None, punctuation)

    stop_words = stopwords.words('english')
    if keep_questions:
        for q_word in ('who', 'what', 'when', 'where', 'why', 'how', 'which', 'can'):
            stop_words.remove(q_word)

    if number_token:
        text = text.replace('NUMBERTOKEN', '<**num**>')

    if rm_stop_words == 1:
        text = [x.strip() for x in text.split() if x.lower() not in stop_words]
    elif rm_stop_words == -1:
        text = [x.strip() if x.lower() not in stop_words else "<**stop**>" for x in text.split()]  
    else:
        text = [x.strip() for x in text.split()] 

    if stemmer is not None:
        text = [stemmer.stem(x) for x in text]

    if lemmatizer is not None:
        text = [lemmatizer.lemmatize(x) for x in text]

    return text


def unit_tests():
    print('Running Unit Tests')
    assert pipeline('123,456.00') == ['123456.00']
    assert pipeline('123,456.00', number_token=True) == ['<**num**>']
    assert pipeline('123456.00', number_token=True) == ['<**num**>']
    assert pipeline('1245600', number_token=True) == ['<**num**>']
    assert pipeline('123,456.00', number_token=True, rm_punc=True) == ['<**num**>']
    assert pipeline('what123,456.00', number_token=True, rm_punc=True) == ['what', '<**num**>']
    assert pipeline('is', rm_stop_words=1) == []
    assert pipeline('why?', rm_stop_words=1) == ['?']
    assert pipeline('why?', rm_stop_words=1, keep_questions=True) == ['why','?']
    assert pipeline('why?', rm_stop_words=1, keep_questions=True, rm_punc=True) == ['why']
    assert pipeline('why?', rm_stop_words=-1, keep_questions=False, rm_punc=True) == ['<**stop**>']
    assert pipeline('normal', rm_stop_words=1, keep_questions=True, rm_punc=True, number_token=True) == ['normal']
    # lemmatizer = WordNetLemmatizer()
    # assert pipeline('Wolves', lemmatizer=lemmatizer) == ['wolf']
    assert pipeline("When's", rm_stop_words=1, keep_questions=True, rm_punc=True) == ['when']
    assert pipeline("When's", rm_stop_words=1, keep_questions=False, rm_punc=True) == []
    assert pipeline("When's", rm_stop_words=-1, keep_questions=True, rm_punc=True) == ['when', '<**stop**>']
    assert pipeline('"Was"', keep_questions=True, rm_punc=True) == ['was']

    print('All Unit Tests Passed')


if __name__ == '__main__':
    unit_tests()


