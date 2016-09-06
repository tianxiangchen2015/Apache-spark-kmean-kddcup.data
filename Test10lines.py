import pprint
from math import log
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans,KMeansModel
from pyspark.mllib.feature import StandardScaler
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
    max_k = 10
    # load data to SparkContext
    pp = pprint.PrettyPrinter(indent=10)
    raw_data = sc.textFile('kddcup.data_10_percent_corrected')

    # Count all the different labels
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
    path = '/Users/apple/Dropbox/2016Spring/COSC526/homework2/Best'
    # Load the k-means model
    best_model = KMeansModel.load(sc, path)
    label_data = labels.zip(standardized_data_values)
    # Get the last ten lines data
    label_data2 = standardized_data_values.zipWithIndex()
    label_data3 = label_data2.filter(lambda x: x[1] in range(size-10,size,1))
    # Predict and print the result.
    k_labels = label_data3.map(lambda datum: best_model.predict(datum[0]))
    print(k_labels.collect())
    k_labels_all = label_data.map(lambda datum: (best_model.predict(datum[1]),datum[0]))
    result = k_labels_all.groupByKey().mapValues(lambda x: Counter(x)).collect()

    print(labels.zipWithIndex().filter(lambda x: x[1] in range(size-10,size,1)).collect())
    print(k_labels.collect())
    print(result)
