# MIDAS Task 3(NLP)
A natural language processsing model written in python using jupyter notebook.

## Problem
The problem was identified as text classification task in NLP in which product categorisation is being done based on the description of product. The dataset is taken from an e-commerce website flipkart and the link to the dataset can be found [here](https://docs.google.com/spreadsheets/d/1pLv0fNE4WHokpJHUIs-FTVnmI9STgog05e658qEON0I/edit?usp=sharing ). 
The data is split into training and test sets to first train the ML models and then test its accuracy.  **F-1 score** and **LogLoss function** are used as a metric for accuracy.

## Approach
I approached the model in following in the follwing way:
* **Data Pre-processing
* **Fitting the model

### Data Pre-processing
The data preprocessing is started off by cleaning the data. The two useful variables in the data were `product_category_tree` and `description`. Rest of the columns are dropped.

The category is extracted from the `product_category_tree` column. The level 1 of the hierarchy tree is defined as the primary category for our predictions.
The text data first needs to be converted to a numeric representation before ML algorithms are applied to it. The methods which achieve this goal are called text vectorisation methods. The most popular ones are Bag of Words and TF-IDF. In most cases, they result in what is called a Term-Document matrix (TDM). I aslo used glove embeddings.

### Fitting the best model

After the necessary data preprocessing the various ML models were fitted on the data. The F-1 score and multiclass log loss are choosen as a metric to measure the accuracy of the models.




