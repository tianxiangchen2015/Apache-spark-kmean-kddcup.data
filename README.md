# Apache Spark-kmean solve Kddcup cyber-attacts problem

In this project implement an unsupervised data mining approach, “k-means” algorithm, as a means to detect cyber-attacks. In the previous assignment, we have already implemented a classifier based on “known patterns”. But what about unknown patterns? The k-means algorithm can cluster network connections based on the statistics about each of them. Thus, in this assignment, we will use dataset provided by KDD cup (http://www.kaggle.com) to test our k-mean algorithm. The dataset is about 700MB and contains information about the raw network packet data. 

## Introduction

K-means clustering is one simple unsupervised learning algorithm that solve the well known clustering problem. The main idea of K-means clustering is to partition the observations of the n-by-p data matrix X into k clusters. The first step is to define k centers for each clusters, and the k centers should be as far as possible from each others. The second step is to allocate each point to the nearest center, when no point is “single”, a prototype model is done. The next step is to calculate k new centers as barycenter of the clusters based on the early model, and reconnect each point to the new k centers. Thus, a loop has been generated. After each iteration, the location of the center has been changed until the centers reach the best locations. 


## Prerequisities

Required softwares:
```
[Apache Spark](http://spark.apache.org)
```

## Dataset
```
[KDD CUPP-1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
```
## Approch



## Training and testing



## Author

* **Tianxiang Chen (ORNL Research Assistant)** - [Linkedin HomePage](https://www.linkedin.com/in/tianxiang-chen-946543114?trk=nav_responsive_tab_profile)


## Acknowledgments

* **jadianes/kdd-cup-99-spark** -[Github Repository](https://github.com/jadianes/kdd-cup-99-spark)
    
