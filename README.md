# MIDAS Task 3(NLP)
A natural language processsing model written in python using jupyter notebook.

## Problem
The problem was identified as text classification task in NLP in which product categorisation is being done based on the description of product. The dataset is taken from an e-commerce website flipkart and the link to the dataset can be found [here](https://docs.google.com/spreadsheets/d/1pLv0fNE4WHokpJHUIs-FTVnmI9STgog05e658qEON0I/edit?usp=sharing ). 
The data is split into training and test sets to first train the ML models and then test its accuracy.  **F-1 score** and **LogLoss function** are used as a metric for accuracy.

## Approach
I approached the model in following in the follwing way:
* Data Pre-processing
* Fitting the model

### Data Pre-processing
The data preprocessing is started off by cleaning the data. The two useful variables in the data were `product_category_tree` and `description`. Rest of the columns are dropped.

The category is extracted from the `product_category_tree` column. The level 1 of the hierarchy tree is defined as the primary category for our predictions.
The data is split into 85% training and 15% test sets.  The models are trained on training sets and then validated on test sets.

Only 24 categories were used from the available 32. The categories that were drop contained less than 25 rows(products). These were removed to because because of lack fo data.

The data cleaning involved removing all the punctuations, splitting the sentences into words and removing all the stopwords.
The text data first needs to be converted to a numeric representation before ML algorithms are applied to it. The methods which achieve this goal are called text vectorisation methods. The most popular ones are TF-IDF and count vectorizer. In most cases, they result in what is called a Term-Document matrix (TDM). I aslo used glove embeddings([link](http://www-nlp.stanford.edu/data/glove.840B.300d.zip ))

### Fitting the best model

After the necessary data preprocessing the various ML models were fitted on the data. The F-1 score and multiclass log loss are choosen as a metric to measure the accuracy of the models.

* Text vectorization using TF-IDF on some classification models

| Metric  | Logistic Regression | SVM          | XGBoost|
| ------------- | ------------- | -------------|-------------|
| Log Loss  |  0.242 | 0.154 | 0.079 |
| F-1 Score  | 0.97  | 0.96  | 0.98 |

* Text vectorization using Count Vectorizer on some classification models

| Metric  | Logistic Regression | SVM          | XGBoost|
| ------------- | ------------- | -------------|-------------|
| Log Loss  | 0.086  | 0.224 | 0.074 |
| F-1 Score  |  0.98 | 0.93  | 0.98 |


We see that the predictions using Count Vectorizer are generally better in comparison to TF-IDF vectorizer. This is because CountVectorizer counts the number of times a word appears in the document which results in biasing in favour of most frequent words. In comparison TfidfVectorizer weights the word counts by a measure of how often they appear in the documents. So Count Vectorizer ends up in ignoring rare words which could have helped is in processing our data more efficiently. But TfidfVectorizer considers overall document weightage of a word. It helps us in dealing with most frequent words. Using it we can penalize them. 

So the most frequent words that may occur in `description` are given equal weights to the rare words in Count Vectorizer. So the most frequent words determine the accuracy of the models.

We see that XGBoost is clearly a better performing model with better accuracy than any other models when used on both techniques of vectorization. 

* Using stopwords
Then the approach was to try using custom stopwords that included the words from the data that were found to be not useful. The data was trained on only logistic regresion model.

| Metric  | TF-IDF | Count vectorizer         |
| ------------- | ------------- | -------------|
| Log Loss  | 0.268  | 0.082 | 
| F-1 Score  | 0.97  | 0.98  | 

There is no significant affect on the accuracy by the use of custom stopwords. So this proves to be not useful. 

* Glove Embeddings
Finally the neural networks were introduced. The nueral networks were trained on glove vectors. 

First the XGBoost was used on vocanulary built using glove vectors.

| Metric  | XGBoost | 
| ------------- | ------------- | 
| Log Loss  | 0.117  |  
| F-1 Score  | 0.97  | 

There were three types of neural networks used - a simple one and a Bi-LSTM and a GRU model.
All of them are trained with glove embeddings.

| Metric  | NN | Bi-LSTM | GRU |
| ------------- | ------------- | -------------|----------|
| Log Loss  | 0.094  | 0.133 | 0.123 |
| F-1 Score  | 0.98  | 0.97 | 0.97 |

<div align="center"> Neural Network training loss vs Validation Loss over number of epochs</div><br />

![NN](https://user-images.githubusercontent.com/47393872/114260264-5e1c1980-99f1-11eb-970d-da229a7f09b4.png)

<div align="center"> BiLSTM model training loss vs Validation Loss over number of epochs</div><br />

![Bilstm](https://user-images.githubusercontent.com/47393872/114260291-83a92300-99f1-11eb-915c-b4fe97a9bb98.png)


<div align="center"> GRU model training loss vs Validation Loss over number of epochs</div><br />

![GRU](https://user-images.githubusercontent.com/47393872/114260298-899f0400-99f1-11eb-966d-a418c599cb28.png)
<br />


We see that as the models progress through epochs the two curves, the training loss curve and the validation loss curve start to converge. This means that with the number of epochs the models are fitting the training and test sets effienctly. 

![NN](https://user-images.githubusercontent.com/47393872/114262920-2c5e7f00-9a00-11eb-9e4b-053fdc8f8146.png)
![Bilstm](https://user-images.githubusercontent.com/47393872/114262921-2d8fac00-9a00-11eb-8000-90529d9b6140.png)
![GRU](https://user-images.githubusercontent.com/47393872/114262923-2d8fac00-9a00-11eb-8724-5b2a0c7f6ad2.png)



## Conclusion
After testing 12 different model combinations on our dataset and evaluating the predictions with metrics of log loss and F1 score we see that XGBoost used on the count vectorizer and TFIDF vectorizer processed data was the best performing. The neural net models are all giving great results when used with glove embeddings. The basic 3 layer neural net is the best performing. This even outperforms the BILSTM and GRU models.



