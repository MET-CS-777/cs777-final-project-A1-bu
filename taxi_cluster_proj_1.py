from __future__ import print_function

import os
import sys
import requests
from operator import add

from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *

# ML library 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors


# Exception Handling and removing wrong datalines
def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


# Data cleaning 
def correctRows(p):
    if len(p) == 17:
        if isfloat(p[5]) and isfloat(p[16]) and isfloat(p[6]) and isfloat(p[7]):
            # Thresholds - fare > 0, distance > 0, valid coordinates
            fare = float(p[11])
            trip_distance = float(p[5])
            total_amount = float(p[16])
            pickup_long = float(p[6])
            pickup_lat = float(p[7])
            tip_amount = float(p[14])
            
            # Basic validation
            if (fare > 0 and trip_distance > 0 and total_amount > 0 and tip_amount >= 0 and
                -74.5 <= pickup_long <= -73.5 and 40.4 <= pickup_lat <= 41.0):
                return p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>", file=sys.stderr)
        exit(-1)
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Taxi Pickup Clustering Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    sc = spark.sparkContext
    sc.setLogLevel("WARN")  # Reduce log verbosity
    
    
    # Read and clean data
    print("Reading and cleaning data...")
    rdd = sc.textFile(sys.argv[1])
    
    # Map rows and filter invalid ones
    rows = rdd.map(lambda line: line.split(','))
    valid_rows = rows.filter(correctRows).cache()  # Cache for reuse
    
    print(f"Total valid rows: {valid_rows.count()}")
    print("Sample valid rows:")
    for row in valid_rows.take(2):
        print(row[:8])  # Print first 8 columns for readability
    
    # Create DataFrame with pickup coordinates
    print("Creating DataFrame...")
    pickup_data = valid_rows.map(lambda x: Row(
        pickup_longitude=float(x[6]),
        pickup_latitude=float(x[7]),
        fare_amount=float(x[16]),
        trip_distance=float(x[5])
    ))
    
    df = spark.createDataFrame(pickup_data).cache()
    print(f"DataFrame created with {df.count()} rows")
    
    # Show basic statistics
    print("Pickup coordinate statistics:")
    df.select("pickup_longitude", "pickup_latitude").describe().show()
    
    # Create feature vector
    print("Creating feature vectors...")
    assembler = VectorAssembler(
        inputCols=["pickup_longitude", "pickup_latitude"],
        outputCol="features"
    )
    feature_df = assembler.transform(df).cache()  

    # Split Data
    train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set count: {train_df.count()}")
    print(f"Testing set count: {test_df.count()}")
    
    # Use predetermined k value
    k = 19
    print(f"Using k={k} clusters")
    
    # Train KMeans model
    print("Training KMeans model...")
    kmeans = KMeans(
        featuresCol="features", 
        predictionCol="prediction", 
        k=k, 
        seed=42,
        maxIter=20,
        tol=1e-4
    )
    
    model = kmeans.fit(train_df)
    
    # Make predictions
    print("Making predictions...")
    clustered_df = model.transform(test_df).cache()
    
    # Print cluster centers
    print(f"\nCluster Centers for k={k}:")
    centers = model.clusterCenters()
    for i, center in enumerate(centers):
        print(f"Cluster {i}: Longitude={center[0]:.6f}, Latitude={center[1]:.6f}")
    
    # Show sample predictions
    print("\nSample clustered points:")
    clustered_df.select("pickup_longitude", "pickup_latitude", "prediction", "fare_amount").show(10)
    
    # Cluster analysis
    print("\nCluster Analysis:")
    cluster_stats = clustered_df.groupBy("prediction") \
        .agg(
            func.count("*").alias("point_count"),
            func.round(func.avg("fare_amount"),2).alias("avg_fare"),
            func.round(func.avg("trip_distance"),2).alias("avg_distance")
        ).orderBy("prediction")
    
    cluster_stats.show()
    
    # Evaluate clustering quality
    print("Evaluating clustering quality...")
    evaluator = ClusteringEvaluator(
        featuresCol="features", 
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )
    
    silhouette_score = evaluator.evaluate(clustered_df)
    print(f"Silhouette Score: {silhouette_score:.4f}")
    
    # Training cost (WSSSE)
    wssse = model.summary.trainingCost
    print(f"Within Set Sum of Squared Errors: {wssse:.2f}")
    
    print("Analysis completed successfully!")
    
    spark.stop()

