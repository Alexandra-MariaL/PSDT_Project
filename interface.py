import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import PySimpleGUI as sg

pd.options.mode.chained_assignment = None  # default='warn'

# **********************************************************************************************************************
# Input Data Preprocessing

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

# **********************************************************************************************************************
# Logistic Regression Model

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

test.reset_index(drop=True, inplace=True)  # resetam indexul pentru a itera
# calculam acuratetea
accuracy = 0
for i in range(len(test)):
    if test["Sentiment"][i] == predictions[i]:
        accuracy += 1
    # print(test["Phrase"][i] + "\nactual: " + test["Sentiment"][i] + "\npredicted: " + predictions[i] + "\n")
accuracy /= len(test)
accuracy = round(accuracy, 2)
# print("accuracy: " + str(accuracy))

# **********************************************************************************************************************
# Interfata

layout = [
    [
        sg.Text("Search keyword:"),
        sg.In(size=(40, 1), enable_events=True, key="-KEYWORD-"),
        sg.Text("                          "),
        sg.Button("Classify"),
        sg.VSeperator(),
        sg.Text("Result of classification:"),
        sg.Text(size=(40, 1), key="-OUT-MESSAGE-")
    ],
    [sg.HSeparator()],
    [
        sg.Text("Or pick one from the list below:"),
        sg.Text("                                                                                                  "),
        sg.Text("Logistic Regression Model accuracy: " + str(accuracy * 100) + "%")
    ],
    [
        sg.Listbox(
            values=test["Phrase"], enable_events=True, size=(150, 25), key="-REVIEWS-LIST-"
        )
    ]
]

window = sg.Window("Sentiment Analysis", layout)
g_review = ""  # aceasta variabila va stoca textul recenziei selectate din listbox

# executa bucla evenimentelor
while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # butonul "Classify" a fost apasat
    if event == "Classify":
        if g_review != "":
            for i in range(len(test)):
                if test["Phrase"][i] == g_review:
                    out_message = predictions[i] + " (actual: " + test["Sentiment"][i] + ")"
                    window["-OUT-MESSAGE-"].update(out_message)
                    break
        else:
            print("Please select a review first!")

    # o recenzie a fost selectata din listbox
    if event == "-REVIEWS-LIST-":
        window["-OUT-MESSAGE-"].update("")  # resetam mesajul clasificarii
        # asignam variabilei globale g_review textul recenziei selectate din listbox
        g_review = values["-REVIEWS-LIST-"][0]
        print("\nSelected review:\n" + g_review)

    # a fost adaugat un cuvant cheie
    if event == "-KEYWORD-":
        keyword = values["-KEYWORD-"]
        print("\nFollowing reviews contains the keyword " + keyword + ":")
        # cautam si afisam toate recenziile care contin cuvantul cheie adaugat
        for i in range(len(test)):
            if test["Phrase"][i].__contains__(keyword):
                print(test["Phrase"][i])

window.close()
