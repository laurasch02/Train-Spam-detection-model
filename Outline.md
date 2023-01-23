_Used repositories:_
  - pandas  --> for data cleaning and analysis
  - scikit-learn --> machine learning tool for classifying the data

_preprocessing_:
  - preprocess the data from the spam.csv file 
  - split into training and test data for the model
  - with the method CountVector() count the number of word occurences in each email

_building the model_:
  - using the support vector machine algorithm from sklearn to classify the data
  - the model will create a decision boundary on which it classifies the data from our CountVector() method
 
 _testing the model_:
  - fitting the the test data with CountVectorizer()
  - print out the Accuracy of the model

