import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':

    # TODO Step 1: Convert from JSON to dataframe.
    df = pd.read_json("train.json")
    # print("df.info():")
    # df.info()

    # TODO Step 2: Data Analysis (Vlad & Catalin)
    # Citim polaritatile din dataframe.
    polarities = []  # lista care va stoca polaritatile din dataframe

    for i in range(len(df["opinions"])):  # pentru toate review-urile din dataframe
        for j in range(len(df["opinions"][i])):  # pentru fiecare opinie a unui review
            polarities.append(df["opinions"][i][j]["Polarity"])  # adauga in lista polaritatea opiniei curente

    # Observam variabila "Polarity" pentru a vedea dacă majoritatea evaluărilor clienților
    # sunt pozitive sau negative.
    print("\npolarities: " + str(polarities))

    # Cream si afisam in browser o histograma pe baza polaritatilor.
    color = sns.color_palette()
    fig = px.histogram(df, x=polarities)
    fig.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
    fig.update_layout(title_text='Review Polarity')
    # fig.show()

    # Calculam numărul de polarități pozitive și numărul de polarități negative.
    pos = 0  # variabila care stocheaza numarul de polaritati pozitive
    neg = 0  # variabila care stocheaza numarul de polaritati negative

    for i in range(len(polarities)):  # parcurgem polaritatile
        if polarities[i] == "Positive":
            pos = pos + 1
        elif polarities[i] == "Negative":
            neg = neg + 1

    total = len(polarities)
    pos_percent = round(pos * 100 / total, 2)  # procentul polaritatilor pozitive
    neg_percent = round(neg * 100 / total, 2)  # procentul polaritatilor negative

    print("\nPositive polarities: " + str(pos) + "/" + str(total) + " = " + str(pos_percent) + "%")
    print("Negative polarities: " + str(neg) + "/" + str(total) + " = " + str(neg_percent) + "%")

    # Concluzie: aceste cuvinte sunt în cea mai mare parte pozitive, indicând, de asemenea, că majoritatea
    # recenziilor din setul de date exprimă un sentiment pozitiv.

    # TODO Step 3: Classifying Reviews (Alexandra & Dan)
    # Opiniile sunt clasificate în pozitive și negative în funcție de polaritate.
    # Clasificam review-urile in functie de opinii si de intensitatile lor.

    # Cream trei liste: una cu review-uri pozitive, una cu review-uri negative si una cu review-uri neutre.
    positive_reviews = []  # lista care va contine review-uri pozitive
    negative_reviews = []  # lista care va contine review-uri negative
    neutral_review = []  # lista care va contine review-uri neutre

    review_score = 0  # variabila care va determina tipul unui review

    for i in range(len(df["opinions"])):
        for j in range(len(df["opinions"][i])):
            if df["opinions"][i][j]["Polarity"] == 'Positive':
                if df["opinions"][i][j]["Intensity"] == 'Standard':
                    review_score = review_score + 1
                elif df["opinions"][i][j]["Intensity"] == 'Strong':
                    review_score = review_score + 2
            elif df["opinions"][i][j]["Polarity"] == 'Negative':
                if df["opinions"][i][j]["Intensity"] == 'Standard':
                    review_score = review_score - 1
                elif df["opinions"][i][j]["Intensity"] == 'Strong':
                    review_score = review_score - 2

        if review_score > 0:
            positive_reviews.append(df["text"][i])
        elif review_score < 0:
            negative_reviews.append(df["text"][i])
        elif review_score == 0:
            neutral_review.append(df["text"][i])

        review_score = 0

    positive_df = pd.DataFrame(positive_reviews, columns=["text"])
    negative_df = pd.DataFrame(negative_reviews, columns=["text"])
    neutral_df = pd.DataFrame(neutral_review, columns=["text"])

    print("\nPositive reviews:" + str(positive_df))
    print("\nNegative reviews:" + str(negative_df))
    print("\nNeutral reviews:" + str(neutral_df))

    # TODO Step 4: More Data Analysis (Paula)

    # Observam cele mai utilizate cuvinte din cele 3 dataframe-uri.
    positive_words_frequencies = positive_df["text"].str.split().explode().value_counts()
    negative_words_frequencies = negative_df["text"].str.split().explode().value_counts()
    neutral_words_frequencies = neutral_df["text"].str.split().explode().value_counts()

    print("Some relevant frequently positive words used: ")
    for i in range(50):
        if len(positive_words_frequencies.keys()[i]) > 3:
            print(str(positive_words_frequencies.keys()[i]) + " : " + str(positive_words_frequencies[i]))

    print("Some relevant frequently negative words used: ")
    for i in range(50):
        if len(negative_words_frequencies.keys()[i]) > 3:
            print(str(negative_words_frequencies.keys()[i]) + " : " + str(negative_words_frequencies[i]))

    print("Some relevant frequently neutral words used: ")
    for i in range(50):
        if len(neutral_words_frequencies.keys()[i]) > 3:
            print(str(neutral_words_frequencies.keys()[i]) + " : " + str(neutral_words_frequencies[i]))

    stopwords = set(STOPWORDS)
    stopwords.update(["hotel", "room", "restaurant", "breakfast", "stay", "location", "service", "staff", "food", "bar",
                      "holiday", "restaurants", "rooms", "place", "city", "area", "time", "day", "bathroom", "night",
                      "one", "floor", "go", "don", "t", "s", "beach", "pool", "will"])

    # Cream si afisam o figura cu 'nori de cuvinte' din positive_df.
    positive_words = " ".join(review for review in positive_df["text"])
    wordcloud1 = WordCloud(stopwords=stopwords).generate(positive_words)
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.axis("off")
    plt.title("positive_df wordcloud")
    plt.show()

    # Cream si afisam o figura cu 'nori de cuvinte' din negative_df.
    negative_words = " ".join(review for review in negative_df["text"])
    wordcloud2 = WordCloud(stopwords=stopwords).generate(negative_words)
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")
    plt.title("negative_df wordcloud")
    plt.show()

    # Cream si afisam o figura cu 'nori de cuvinte' din neutral_df.
    neutral_words = " ".join(review for review in neutral_df["text"])
    wordcloud3 = WordCloud(stopwords=stopwords).generate(neutral_words)
    plt.imshow(wordcloud3, interpolation='bilinear')
    plt.axis("off")
    plt.title("neutral_df wordcloud")
    plt.show()

    # TODO Step 5: Building the Model

    # Data Cleaning: eliminam caracterele speciale din fiecare dataframe.
    # Definim o functie care va elimina caracterele speciale.
    def remove_punctuation(text):
        final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', '(', ')'))
        return final

    positive_df['text'] = positive_df['text'].apply(remove_punctuation)
    negative_df['text'] = negative_df['text'].apply(remove_punctuation)
    neutral_df['text'] = neutral_df['text'].apply(remove_punctuation)

    # Coloana 'polarity' din positive_df este populata cu len(positive_df) valori de 'positive'.
    positive_df['polarity'] = ['positive' for i in range(len(positive_df))]
    # Coloana 'polarity' din negative_df este populata cu len(negative_df) valori de 'negative'.
    negative_df['polarity'] = ['negative' for i in range(len(negative_df))]
    # Coloana 'polarity' din neutral_df este populata cu len(neutral_df) valori de 'neutral'.
    neutral_df['polarity'] = ['neutral' for i in range(len(neutral_df))]

    # Concatenam toate dataframe-urile.
    df = pd.concat([positive_df, negative_df, neutral_df], ignore_index=True)

    # Impartim datele in 80% date pentru antrenament si 20% date pentru testare.
    index = df.index

    # Adaugam o noua coloana (random_number) in df si o populam cu valori aleatorii.
    df['random_number'] = np.random.randn(len(index))
    train = df[df['random_number'] <= 0.8]
    test = df[df['random_number'] > 0.8]

    # print("\ntrain.info():")
    # train.info()
    # print("\ntest.info():")
    # test.info()

    # Vectorizam dataframe-urile test si train.
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(train['text'])
    test_matrix = vectorizer.transform(test['text'])

    # Aplicam regresia logistica.
    lr = LogisticRegression()

    # Definim variabilele de antrenare si testare.
    X_train = train_matrix  # contine vectori de cuvinte din datele de antrenare
    X_test = test_matrix  # contine vectori de cuvinte din datele de testare
    y_train = train['polarity']  # contine coloana polarity din datele de antrenare
    y_test = test['polarity']  # contine coloana polarity din datele de testare

    # Antrenam modelul astfel incat sa functioneze pe date de intrare noi (de test).
    lr.fit(X_train, y_train)

    # Facem predictii avand ca intrare date de test (review-uri cu care nu a fost antrenat modelul).
    predictions = lr.predict(X_test)

    # TODO Step 6: Testing

    # Afisam matricea de confuzie.
    new = np.asarray(y_test)
    confusion_matrix = confusion_matrix(predictions, y_test)
    print("\nconfusion_matrix:")
    print(confusion_matrix)

    # Afisam acuratetea modelului.
    print("\nclassification_report:")
    print(classification_report(predictions, y_test))
