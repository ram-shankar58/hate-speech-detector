import nltk

def stopwords():
    stopwords=nltk.corpus.stopwords.words("english")

    exclusions=["#ff", "ff", "rt", "austrian painter"] #add any words you want to ignore
    stopwords.extend(exclusions)
    return stopwords