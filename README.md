# Apache Spark-kmean solve Kddcup cyber-attacts problem

In this project implement an unsupervised data mining approach, “k-means” algorithm, as a means to detect cyber-attacks. In the previous assignment, we have already implemented a classifier based on “known patterns”. But what about unknown patterns? The k-means algorithm can cluster network connections based on the statistics about each of them. Thus, in this assignment, we will use dataset provided by KDD cup (http://www.kaggle.com) to test our k-mean algorithm. The dataset is about 700MB and contains information about the raw network packet data. 

## Introduction

In this assignment, we will train a Naïve Bayes classification using “Reuters-21578” dataset. It is currently the most widely used test collection for text categorization research. The data was originally collected by Reuters and is already been labeled. The datasets contains 22 .sgm documents and each files contains approximately 1000 papers in various topics. In this assignment we will only focus on the following topics: “money, fx, crude, grain, trade, interest, wheat, ship, corn, oil, dlr, gas, oilseed, supply, sugar, gnp, coffee, veg, gold, soybean, bop, livestock, cpi.”.  We will parse these XML files and get the papers of interested topics, and create TF-IDF dictionary to train the Naïve Bayes. 

## Prerequisities

Required softwares:
```
[Apache Spark](http://spark.apache.org)
[Nature Language Toolkit](http://www.nltk.org)
[goose extractor](https://pypi.python.org/pypi/goose-extractor/)
```

## Dataset
```
[Reuters-21578](http://www.daviddlewis.com/resources/testcollections/reuters21578/)
```
## Approch

1. Parse the XML documents using python build in “sgml” library, extract all the topics and body and put them in to a python list “docs= [(topic, body)]”. Then, filter the list and get only the interested topics. Also, eliminate all the dirty data, such as documents with an empty body
2. Tokenize and stem the body part using Nature Language Processing Toolkit (NLTK) so that we can only keep the meaningful content. In this part, we use Lancast Stemmer to stem all the words.
3. Vectorizing the documents and TF-IDF calculation. Load the documents into Spark Context, which is a resilient distributed dataset (RDD). RDD is a collection of elements partitioned across the nodes of the cluster that can be operated on in parallel. After load the data into Spark, implement a vector representation of the documents using TF-IDF method. Term frequency  is the number of times that term  appears in document . And IDF is the inverse document frequency
4. Training and testing Naïve bayes model. Randomly split all the data into two parts – 60% of data as training data and 40% as testing data. 

## Training and testing

1. parse.py -- parsing and cleaning the raw reuters-21578 data(XML). Then save the clean data into train_data_final.txt.
2. trainNaiveBayes.py -- Calculate TF-IDF. Train and test the naive bayes model.
3. xmlOnline.py -- Extract text online to test our bayes model.

## Author

* **Tianxiang Chen (ORNL Research Assistant)** - [Linkedin HomePage](https://www.linkedin.com/in/tianxiang-chen-946543114?trk=nav_responsive_tab_profile)


## Acknowledgments

* Use sample code from https://chimpler.wordpress.com/2014/06/11/classifiying-documents-using-naive-bayes-on-apache-spark-mllib/
* 
    
