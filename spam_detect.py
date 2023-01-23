from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pandas as pd


spam = pd.read_csv('spam.csv')

spam = spam.rename(columns=({"v1": "Labels", "v2": "Email Text"}))

labels = spam["Labels"]
text = spam["Email Text"]

labels_train, labels_test, text_train, text_test = train_test_split(labels, text, test_size=0.2)

### randomly assign number to tokenized word and count occurences ###
cv = CountVectorizer()
features = cv.fit_transform(text_train)

### building the model ###

model = svm.SVC()
model.fit(features, labels_train)

features_test = cv.transform(text_test)
print(f"Accuracy: {model.score(features_test, labels_test)}")