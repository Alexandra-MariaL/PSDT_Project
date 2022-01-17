import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

pd.options.mode.chained_assignment = None  # default='warn'

# **************************************** Prelucrarea datelor de intrare ****************************************


# Tokenization (despartim string-urile intr-o lista de cuvinte)
def sentence_tokenize(text):
    return nltk.sent_tokenize(text)


def word_tokenize(text):
    # returneaza lista de cuvinte
    return nltk.word_tokenize(text)


# ne asiguram ca fiecare litera este lowercase
def lower_words(text):
    return text.lower()


# eliminam numerele
def remove_numbers(text):
    output = ''.join(c for c in text if not c.isdigit())
    return output


def remove_punctuation(text):
    return ''.join(c for c in text if c not in punctuation)


# eliminam cuvintele irelevante
def remove_stopwords(sentence):
    stop_words = stopwords.words("english")
    return ' '.join([w for w in nltk.word_tokenize(sentence) if w not in stop_words])


# normalizare
def lemmatize(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]
    return " ".join(lemmatized_word)


# punem toate functiile de curatare intr-o singura functie
def preprocess(text):
    lower_text = lower_words(text)
    sentence_tokens = sentence_tokenize(lower_text)
    word_list = []
    for each_sent in sentence_tokens:
        lemmatizzed_sent = lemmatize(each_sent)
        clean_text = remove_numbers(lemmatizzed_sent)
        clean_text = remove_punctuation(clean_text)
        clean_text = remove_stopwords(clean_text)
        word_tokens = word_tokenize(clean_text)
        for wt in word_tokens:
            word_list.append(wt)
    return word_list


train_data = pd.read_json("train.json")  # conversie de la json la dataframe
train_data.drop("sent_id", axis="columns", inplace=True)  # eliminarea coloanei "sent_id"
train_data.rename(columns={"text": "Phrase"}, inplace=True)  # redenumirea coloanei "text" in "Phrase"
train_data.insert(2, "Sentiment", "", True)  # inserarea coloanei noi "Sentiment"

# clasificam recenziile in pozitive si negative
# calcularea valorilor coloanei "Sentiment" pe baza coloanei "opinions"
review_score = 0
for i in range(len(train_data["Phrase"])):
    if len(train_data["opinions"][i]) > 0:
        for j in range(len(train_data["opinions"][i])):
            if train_data["opinions"][i][j]["Polarity"] == "Positive":
                if train_data["opinions"][i][j]["Intensity"] == "Standard":
                    review_score = review_score + 1
                elif train_data["opinions"][i][j]["Intensity"] == "Strong":
                    review_score = review_score + 2
            elif train_data["opinions"][i][j]["Polarity"] == "Negative":
                if train_data["opinions"][i][j]["Intensity"] == "Standard":
                    review_score = review_score - 1
                elif train_data["opinions"][i][j]["Intensity"] == "Strong":
                    review_score = review_score - 2
        if review_score > 0:
            train_data["Sentiment"][i] = "Positive"
        else:
            train_data["Sentiment"][i] = "Negative"
        review_score = 0
    else:
        train_data["Sentiment"][i] = "None"

# eliminam recenziile fara opinii
train_data.drop(train_data[train_data["Sentiment"] == "None"].index, inplace=True)
train_data.reset_index(drop=True, inplace=True)  # resetam indexul dupa eliminarea randurilor
train_data.drop("opinions", axis="columns", inplace=True)  # eliminam coloana "opinions"

# **************************************** Analiza datelor de intrare ****************************************

train_data.info()  # informatii despre baza de date

# afisam recenziile
# for i in range(len(train_data)):
#     print("\n" + str(i) + " : " + train_data["Phrase"][i] + "\n" + train_data["Sentiment"][i])

print("\nSentiment distribution:\n" + str(train_data["Sentiment"].value_counts()))  # vedem distributia sentimentelor

# afisam in browser o histograma pe baza sentimentelor
color = sns.color_palette()
fig = px.histogram(train_data, x=train_data["Sentiment"])
fig.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
fig.update_layout(title_text='Review Sentiment')
fig.show()

# calculam si afisam procentele sentimentelor pozitive si negative
positive_reviews = []
negative_reviews = []
positive_count = 0
negative_count = 0
total_count = len(train_data)

for i in range(total_count):
    if train_data["Sentiment"][i] == "Positive":
        positive_count += 1
        positive_reviews.append(train_data["Phrase"][i])
    elif train_data["Sentiment"][i] == "Negative":
        negative_count += 1
        negative_reviews.append(train_data["Phrase"][i])

positive_percent = round(positive_count * 100 / total_count, 2)
negative_percent = round(negative_count * 100 / total_count, 2)

print("\nPositive sentiments percent: " + str(positive_percent) + "%")
print("Negative sentiments percent: " + str(negative_percent) + "%")

# Cream si afisam un wordcloud cu recenziile pozitive
positive_words = " ".join(review for review in positive_reviews)
wordcloud_positive = WordCloud().generate(positive_words)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis("off")
plt.title("positive wordcloud")
plt.show()

# Cream si afisam un wordcloud cu recenziile negative
negative_words = " ".join(review for review in negative_reviews)
wordcloud_negative = WordCloud().generate(negative_words)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis("off")
plt.title("negative wordcloud")
plt.show()

# # procesarea datelor de intrare
# for i in range(len(train_data)):
#     train_data["Phrase"][i] = preprocess(train_data["Phrase"][i])
# # afisarea recenziilor procesate (datele de intrare pt modelul pipeline)
# for i in range(len(train_data)):
#     print("\n" + str(i) + " : " + str(train_data["Phrase"][i]) + "\n" + str(train_data["Sentiment"][i]))

# **************************************** Logistic Regression Model ****************************************

index = train_data.index

# adaugam o noua coloana (random_number) in df si o populam cu valori aleatorii
train_data["random_number"] = np.random.randn(len(index))

# impartim datele de intrare in 80% date pentru antrenament si 20% date pentru testare
train = train_data[train_data["random_number"] <= 0.8]
test = train_data[train_data["random_number"] > 0.8]

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')  # vectorizam dataframe-urile test si train
train_matrix = vectorizer.fit_transform(train["Phrase"])
test_matrix = vectorizer.transform(test["Phrase"])

lr = LogisticRegression()  # aplicam regresia logistica

# definim variabilele de antrenare si testare
X_train = train_matrix  # contine vectori de cuvinte din datele de antrenare
X_test = test_matrix  # contine vectori de cuvinte din datele de testare
y_train = train["Sentiment"]  # contine coloana polarity din datele de antrenare
y_test = test["Sentiment"]  # contine coloana polarity din datele de testare

lr.fit(X_train, y_train)  # antrenam modelul astfel incat sa functioneze pe date de intrare noi (de test)

# facem predictii avand ca intrare date de test (review-uri cu care nu a fost antrenat modelul)
predictions = lr.predict(X_test)
confusion_matrix = confusion_matrix(predictions, y_test)  # afisam matricea de confuzie

print("\nLogistic Regression model classification report:")
print(classification_report(predictions, y_test))  # afisam acuratetea modelului

# **************************************** Pipeline (Naive Bayes) Model ****************************************

bow = CountVectorizer(analyzer=preprocess)
classifier = MultinomialNB()
tfidf = TfidfTransformer()

pipeline = Pipeline([
    ("bow", bow),  # bag of words
    ("tfidf", tfidf),  # term frequency–inverse document frequency
    ("classifier", classifier),  # Naive Bayes classifier
])
pipeline.fit(train_data["Phrase"], train_data["Sentiment"])
print("Naive Bayes model accuracy: " + str(round(pipeline.score(train_data["Phrase"], train_data["Sentiment"]), 2)))

# **************************************** Decision Tree Model ****************************************

# cream obiectul de clasificare bazat pe arbori de decizie
clf = DecisionTreeClassifier()

# antrenam clasificatorul arborelui de decizie
clf = clf.fit(X_train, y_train)

# afisam acuratetea
y_pred = clf.predict(X_test)
print("Decision Tree model accuracy: ", round(metrics.accuracy_score(y_test, y_pred), 2))
