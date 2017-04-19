import re
from string import punctuation

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


#Must return list of words
def pipeline(x, lower, rm_stop_words, punc_token, number_token):
    x = x.lower()

    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    # text = text.replace("what's", "what is")
    # text = text.replace("who's", "who is")
    # text = text.replace("what's", "what is")

    text = text.replace("'s", "")
    text = text.replace("'ve", " have")
    text = rext.replace("can't", "cannot")
    text = text.replace("n't", " not")
    text = text.replace("i'm", "i am")
    text = text.replace("'re", " are")
    text = text.replace("'d", " would")
    text = text.replace("'ll", " will")
    text = text.replace(",", " ")
    # text = text.replace(".", " ")


    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)


    text = text.split(' ')




    #Remove Stop Words
    # if x in stopwords.words('english'):
    #     return