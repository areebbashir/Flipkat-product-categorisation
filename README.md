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
The text data first needs to be converted to a numeric representation before ML algorithms are applied to it. The methods which achieve this goal are called text vectorisation methods. The most popular ones are TF-IDF and count vectorizer. In most cases, they result in what is called a Term-Document matrix (TDM). I aslo used glove embeddings([0link](http://www-nlp.stanford.edu/data/glove.840B.300d.zip ))

### Fitting the best model

After the necessary data preprocessing the various ML models were fitted on the data. The F-1 score and multiclass log loss are choosen as a metric to measure the accuracy of the models.

Text vectorization using TF-IDF

| Metric  | Logistic Regression | SVM          | XGBoost|
| ------------- | ------------- | -------------|-------------|
| Log Loss  |  0.242 | 0.154 | 0.079 |
| F-1 Score  | 0.97  | 0.96  | 0.98 |

Text vectorization using Count Vectorizer

| Metric  | Logistic Regression | SVM          | XGBoost|
| ------------- | ------------- | -------------|-------------|
| Log Loss  | 0.086  | 0.224 | 0.074 |
| F-1 Score  |  0.98 | 0.93  | 0.98 |

We see that XGBoost is clearly a better performing model with better accuracy than any other models when used on both techniques of vectorization. 

Then I tried using custom stopwords that included the words from the data that I found to be not useful. I trained them on only logistic regresion model but saw no improvement

| Metric  | TF-IDF | Count vectorizer         |
| ------------- | ------------- | -------------|
| Log Loss  | 0.268  | 0.082 | 
| F-1 Score  | 0.97  | 0.98  | 

We see that there is no clear affect on the accuracy by the use of custom stopwords. So this proves to be not useful. 


Finally I used Glove embeddings and trained XGBoost and neural networks with them. 

| Metric  | XGBoost | 
| ------------- | ------------- | 
| Log Loss  | 0.117  |  
| F-1 Score  | 0.97  | 

I used two types of neural networks- a simple one and a Bi-LSTM model.
Both of them are trained with glove embeddings.

| Metric  | NN | Bi-LSTM |
| ------------- | ------------- | -------------|
| Log Loss  | 0.080  | 0.133 | 
| F-1 Score  | 0.98  | 0.97 |




