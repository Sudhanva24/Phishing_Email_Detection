# Phishing_Email_Detection

In this Project We are Using Various Machine Learning Techniques and Large Lanaguage Models to detect Phishing Emails.

## Dataset

- The dataset consists of approximately 18650  data points.
- It is divided into three subsets: 
  - 14920 data points for training,
  - 1865 data points for validation,
  - 1865 data points for testing.
- The data includes various features relevant to identifying phishing emails.
- my training data consisted of about 9000 Safe and 6000 phishing Emails so we can say the data is slightly imbalanced 
- Each data point is labeled as either 'Phishing Email' or 'Safe Email' to facilitate supervised learning.
- The dataset is balanced to ensure effective training of machine learning models.

## Exploratory Data Analysis

-We have perfomed EDA on the train set to avoid data leakage
- We have analysed various feartures like-:
     - Word Length 
     - Dollar Count
     - Excalamtion Count
     - Topic Modelling Analysis
     - Has_email, Has_ph_no
     - Diacritic Counts
- Refer to the EDA notebook for detailed graphs and overviews

## Feature Engineering

- This is the Heart of Data science 
- After thinking of various features in EDA we now wanted solid evidence to back them
- Applied Statistical Tests like t-test , Chi-square test to see if the features were statisticallly Significant
- Used Mutual Information Scores to decide which features are contributing in reducing uncertaininty in the features
- Also used spearman coefficient to check for non-linear correlation between the features
- Then my trump card for statistical modelling for topic modelling using NMF factorization with the tfidf vectorizer
- Divided the corpus into 10 different topics based on the words occuring in the documents
- There Were about 11 empty emails which all were classified as phsihing, so we decided to replace the Null values by an empty string itslef
- The dominant topic matrix had returuned numerical variables which was converted to one-hot encoding to aviod tooic 1 being cosidered inferior to topic 5
- We also cleaned the text by removing punctuations(except ! and $), stop_words,URL's, links, Ph.no, emails.
- Ph.no and emails were not helpful in predicting Phishing and Safe emails based on the Mutual Information Scores
- Finally used to pipeline to collect all the feature transformations applied so that we can easily apply them to val and test sets

## Clssicial_ML_Techniques

-We used various ML technqiues like 
       - XGboost
       - Random Forest
       - Naive Bayes

- The Tree algorithms were applied to the extracted Features and not the text 
- We applied Naive bayes to clean_text 
- We used tfidf vectorizer along with naive bayes to weigh the probabilities so that words which are very frequent in all the documents are weighed less than those specific to a document

Below I have attached the Scores of How well the Algorithms faired. These are obtained via logging models on Mlflow 

![alt text](<accuracy .png>)

I have attached the F1 scores and Recall scores for Phishing class as this matters. Technically we want that out of 100 phshing emails we capture the maximum of them

![alt text](<Screenshot 2025-02-04 at 10.14.58 PM.png>)

Let us see it pictorially 

![alt text](<Screenshot 2025-02-04 at 10.14.37 PM.png>)

- We can see that Naive Bayes achivies good performence

-But still the feature engineering helped us undertanding the data more 

## Transformers Model

- Finally we also applied DistilBert model. We finetuned the whole model 
- As expected we got excellent results . 
![alt text](<Screenshot 2025-02-05 at 10.46.54 AM.png>)

- But we got a almost as good performance using Naive bayes. 
- So if cost is a constraint Naive bayes is always a good option

