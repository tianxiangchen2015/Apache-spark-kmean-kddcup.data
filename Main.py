import matplotlib.pyplot as plt
import numpy as np
import pprint
from operator import add
from math import log
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans,KMeansModel
from pyspark.mllib.feature import StandardScaler
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
from numpy import array
from math import sqrt

def parse_interaction(line):
    # Parses the data line by line.
    line_split = line.split(",")
    clean_line_split = [line_split[0]]+line_split[4:-1]
    return (line_split[-1], array([float(x) for x in clean_line_split]))


def dist_to_centroid(datum, clusters):
    # Determines the distance of a point to its cluster centroid
    cluster = clusters.predict(datum)
    centroid = clusters.centers[cluster]
    return sqrt(sum([x**2 for x in (centroid - datum)]))


def clustering_score(data, k):
    # Train K-means model and calculate the mean distance
    clusters = KMeans.train(data, k, maxIterations=10, runs=5, initializationMode="random")
    result = (k, clusters, data.map(lambda datum: dist_to_centroid(datum, clusters)).mean())
    print "Clustering score for k=%(k)d is %(score)f" % {"k": k, "score": result[2]}
    return result

def cluster_entropy(result,size):
    # Calculate the Entropy
    H_scale = 0
    size = float(size)
    for x in result:
        items = x[1].values()
        total = float(sum(x[1].values()))
        weight = total/size
        H_i = 0
        for y in items:
            pi = y/total
            H_i += (-pi*log(pi, 2))
        H_scale += (H_i*weight)
    return H_scale

if __name__ == "__main__":

    sc = SparkContext()
    max_k = 180
    # load raw data
    pp = pprint.PrettyPrinter(indent=10)
    print "Loading RAW data..."
    raw_data = sc.textFile('kddcup.data_10_percent_corrected')

    # count by all different labels and print them decreasingly
    print "Counting all different labels"
    labels = raw_data.map(lambda line: line.strip().split(",")[-1])
    size = labels.count()
    print(size)
    label_counts = labels.countByValue()
    sorted_labels = OrderedDict(sorted(label_counts.items(), key=lambda t: t[1], reverse=True))
    for label, count in sorted_labels.items():
        print label, count

    # Prepare data for clustering input
    parsed_data = raw_data.map(parse_interaction)
    parsed_data_values = parsed_data.values().cache()

    # Standardize data
    standardizer = StandardScaler(True, True)
    standardizer_model = standardizer.fit(parsed_data_values)
    standardized_data_values = standardizer_model.transform(parsed_data_values)

    # Result when K = 2
    clusters_2 = KMeans.train(parsed_data_values, 2, maxIterations=10, runs=5, initializationMode="random")

    # Evaluate values of k from 5 to 180 and store the result
    scores = map(lambda k: clustering_score(parsed_data_values,k),range(5,max_k+1,5))
    scores_sd = map(lambda k: clustering_score(standardized_data_values, k), range(5,max_k+1,5))

    t = np.arange(5,max_k+1,5)
    dist = []
    dist_sd = []
    for x in scores:
        dist.append(x[2])

    for x in scores_sd:
        dist_sd.append(x[2])

    # Plot the result
    plt.plot(t,dist,'g^')
    plt.savefig('unsd.png')
    plt.close()

    plt.plot(t,dist_sd,'g^')
    plt.savefig('sd.png')
    plt.close()

    # Calculate the min distance k
    min_k = min(scores, key=lambda x: x[2])[0]
    min_k_sd = min(scores_sd, key=lambda x: x[2])[0]

    # Save the best model
    best_model = min(scores_sd, key=lambda x: x[2])[1]
    path = '/home/chen/Dropbox/2016Spring/COSC526/homework2/Best'
    best_model.save(sc,path)
    H_scale = []
    label_SD = labels.zip(standardized_data_values)

    #Use entropy to evaluate k-means model
    for k in scores_sd:
        model = k[1]
        k_labels = label_SD.map(lambda datum: (model.predict(datum[1]),datum[0]))
        result = k_labels.groupByKey().mapValues(lambda x: Counter(x)).collect()
        H = cluster_entropy(result,size)
        H_scale.append(H)
        print("Clustering score for k=%(k)d is %(entropy)f" % {"k": k[0], "entropy": H})

    print "Best k value is for unsd %(best_k)d" % {"best_k": min_k}
    print "Best k value is %(best_k)d" % {"best_k": min_k_sd}
    print "Result for un-standardized data"
    print(dist)
    print "Result for standardized data"
    print(dist_sd)

    plt.plot(t, H_scale,'g^')
    plt.savefig('H_k.png')
    plt.close()
    print "Finish!"
