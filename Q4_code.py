import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, explode, avg
from pyspark.sql.functions import percent_rank
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

# initialize spark session
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Movie Recommendation") \
    .config("spark.driver.memory", "10g") \
    .config("spark.local.dir", os.environ['TMPDIR']) \
    .getOrCreate()

# create spark context
sc = spark.sparkContext
# set log level to error
sc.setLogLevel("WARN")

print('-'*100)

# load the data
data= spark.read.csv('/users/acq22vk/com6012/ScalableML/Data/ml-20m/ratings.csv', header=True, inferSchema=True).cache()

# order the data by timestamp in ascending order
ordered_data = data.orderBy('timestamp', ascending=True)
# cache the data
ordered_data = ordered_data.cache()

# define the window specification for ordering data based on timestamp
window_spec = Window.orderBy("timestamp")

# apply the window specification to compute the percentile rank
ranked_data = ordered_data.withColumn("percent_rank", percent_rank().over(window_spec))
ranked_data.show(25, truncate=False)

# define a function to split data based on a given threshold
def split_data(ranked_data, threshold):
    # filter data for training and testing based on the threshold
    training = ranked_data.filter(ranked_data["percent_rank"] < threshold).cache()
    testing = ranked_data.filter(ranked_data["percent_rank"] >= threshold).cache()
    return training, testing

# split data into training and testing sets using different thresholds
training1, testing1 = split_data(ranked_data, 0.4)
training2, testing2 = split_data(ranked_data, 0.6)
training3, testing3 = split_data(ranked_data, 0.8)

# set the base seed for ALS
base_seed = 230124165

# initialize ALS with specified parameters
als1 = ALS(userCol="userId", itemCol="movieId", seed=base_seed, coldStartStrategy="drop")

# define metrics for evaluation
evaluators = [
    RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction"),
    RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction"),
    RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
]

# assign each evaluator to a variable
rmse_evaluator, mse_evaluator, mae_evaluator = evaluators

# to compute metrics
def execute_model(train, test, als_model):
    model = als_model.fit(train)
    predictions = model.transform(test)
    rmse = rmse_evaluator.evaluate(predictions)
    mse = mse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    return rmse, mse, mae

# store metrics for visualization
performance_data = {'Setting1': {'40%': {}, '60%': {}, '80%': {}},
                'Setting2': {'40%': {}, '60%': {}, '80%': {}}}

# ollect metrics for Setting 1
performance_data['Setting1']['40%'] = execute_model(training1, testing1, als1)
performance_data['Setting1']['60%'] = execute_model(training2, testing2, als1)
performance_data['Setting1']['80%'] = execute_model(training3, testing3, als1)

# define ALS model
als2 = ALS(userCol="userId", itemCol="movieId", seed=base_seed, coldStartStrategy="drop", rank=20, maxIter=15, regParam=0.02)
performance_data['Setting2']['40%'] = execute_model(training1, testing1, als2)
performance_data['Setting2']['60%'] = execute_model(training2, testing2, als2)
performance_data['Setting2']['80%'] = execute_model(training3, testing3, als2)

# Function to format and display ALS metrics neatly in one DataFrame
def als_report_df(performance_data):
    rows = []
    for setting, data in performance_data.items():
        for split, metrics in data.items():
            rmse, mse, mae = metrics
            rows.append([split, setting, rmse, mse, mae])
    return pd.DataFrame(rows, columns=["Split", "Setting", "RMSE", "MSE", "MAE"])

als_metrics_df = als_report_df(performance_data)
print("ALS Metrics Table:")
print(als_metrics_df)

#bar plot
def plot_performance(metrics, splits, settings, performance_data, colors):
    plt.figure(figsize=(10, 6))
    bar_width = 0.35  # width of each bar
    n_settings = len(settings)
    total_width = n_settings * bar_width  # total width for bars for one metric at one split
    offset = np.arange(len(splits))  # initial offsets for the splits

    for metric_idx, metric in enumerate(metrics):
        for setting_idx, setting in enumerate(settings):
            # Calculate position for each group of bars per setting
            positions = offset + (setting_idx * bar_width) - (total_width / 2) + (bar_width / 2)
            values = [performance_data[setting][split][metric_idx] for split in splits]
            plt.bar(positions, values, bar_width, label=f'{metric} ({setting})', color=colors[metric])

            # Adding annotations above each bar
            for pos, value in zip(positions, values):
                plt.text(pos, value, f'{value:.2f}', ha='center', va='bottom', color=colors[metric])

    plt.title('Performance of ALS Settings Across Different Splits')
    plt.xlabel('Training Split Percentage')
    plt.ylabel('Metric Values')
    plt.xticks(np.arange(len(splits)), splits)  # set x-ticks to be centered for each group of bars
    plt.legend(title="Metrics and Settings")
    plt.grid(True, which='major')  # Enable grid only on y-axis
    plt.tight_layout()
    plt.savefig('/users/acq22vk/com6012/ScalableML/Output/Q4_figA3.jpg')

# Example data setup
splits = ['40%', '60%', '80%']
metrics = ['RMSE', 'MSE', 'MAE']
colors = {'RMSE': 'cyan', 'MSE': 'brown', 'MAE': 'grey'}
line_styles = {'Setting1': '-', 'Setting2': '--'}
performance_data = {
    'Setting1': {'40%': [1, 2, 3], '60%': [1.1, 2.1, 3.1], '80%': [1.2, 2.2, 3.2]},
    'Setting2': {'40%': [0.9, 1.9, 2.9], '60%': [0.8, 1.8, 2.8], '80%': [0.7, 1.7, 2.7]}
}

plot_performance(metrics, splits, ['Setting1', 'Setting2'], performance_data, colors)


print('-'*100)

# function to perform k means clustering
def k_means_clusteranalysis(training_data, k, base_seed):
    model_inst = als2.fit(training_data)
    user_factors = model_inst.userFactors
    k_means = KMeans(k=k, seed=base_seed, featuresCol='features', predictionCol='cluster')
    clusters = k_means.fit(user_factors).transform(user_factors)
    return clusters

def topClusters(clusters):
    clusterSize = clusters.groupBy('cluster').count().orderBy('count', ascending=False).take(5)
    return [row['count'] for row in clusterSize]

# perform the k-means cluster analysis
clusters1 = k_means_clusteranalysis(training1, 25, base_seed)
clusters2 = k_means_clusteranalysis(training2, 25, base_seed)
clusters3 = k_means_clusteranalysis(training3, 25, base_seed)

# Extract top 5 cluster sizes
clusterSize = {
    '40%': topClusters(clusters1),
    '60%': topClusters(clusters2),
    '80%': topClusters(clusters3)
}


print('Top 5 cluster sizes for the 40% training segment:')
print(clusterSize['40%'])

print('Top 5 cluster sizes for the 60% training segment:')
print(clusterSize['60%'])

print('Top 5 cluster sizes for the 80% training segment:')
print(clusterSize['80%'])

clusterSize = pd.DataFrame(clusterSize , index=['1st Largest', '2nd Largest', '3rd Largest', '4th Largest', '5th Largest'])
print('Cluster Sizes Table:')
print(clusterSize)

#line plot
fig, ax = plt.subplots(figsize=(10, 6))
splits = ['40%', '60%', '80%']
x = np.arange(len(splits))

for i, size in enumerate(['1st', '2nd', '3rd', '4th', '5th']):
    cluster_data = [clusterSize [split][i] for split in splits]
    ax.plot(x, cluster_data, marker='o', linestyle='-', label=f'{size} Largest')
    
    # Adding text annotations for each data point
    for j, value in enumerate(cluster_data):
        ax.text(x[j], value, f'{value}', ha='center', va='bottom')

ax.set_xlabel('Training Size Split')
ax.set_ylabel('Number of Users in Cluster')
ax.set_title('Top 5 Largest Clusters by Training Size Split')
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.legend(title="Cluster Rank", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('/users/acq22vk/com6012/ScalableML/Output/Q4_figB1.jpg')


# funtion to find largest movie cluster
def movies_largest_cluster(train_data, clusters):
    largest_user_cluster = clusters.groupBy('cluster').count().orderBy('count', ascending=False).first()[0]
    user_ids = clusters.filter(clusters.cluster == largest_user_cluster).select('id').rdd.map(lambda r: r[0]).collect()
    movie_ratings = train_data.filter((train_data.userId.isin(user_ids)) & (train_data.rating >= 4))
    movies_largest_cluster = movie_ratings.groupBy('movieId').agg(F.avg('rating').alias('average_rating'))
    return movies_largest_cluster.filter('average_rating >= 4')

movies_largest_cluster_1 = movies_largest_cluster(training1, clusters1)
movies_largest_cluster_2 = movies_largest_cluster(training2, clusters2)
movies_largest_cluster_3 = movies_largest_cluster(training3, clusters3)

movies = spark.read.csv('/users/acq22vk/com6012/ScalableML/Data/ml-20m/movies.csv', header=True, inferSchema=True).cache()

# find top 10 genres from top movies
def top_movies(movies_largest_cluster, movies):
    top_movie_ids = movies_largest_cluster.select('movieId').rdd.map(lambda r: r[0]).collect()
    top_movie_genres = movies.filter(movies.movieId.isin(top_movie_ids))
    genre_counts = top_movie_genres.select(explode(F.split(col('genres'), '\|')).alias('genre')).groupBy('genre').count()
    return genre_counts.orderBy('count', ascending=False).limit(10).collect()

top_genres_1 = top_movies(movies_largest_cluster_1, movies)
top_genres_2 = top_movies(movies_largest_cluster_2, movies)
top_genres_3 = top_movies(movies_largest_cluster_3, movies)

def genres(genres_list, split):
    return pd.DataFrame({
        'Split': [split] * len(genres_list),
        'Genre': [genre[0] for genre in genres_list],
        'Count': [genre[1] for genre in genres_list]
    })

# function to find top movies genres
def top_movies_genres(top_movies, movies_df):
    top_movies_df = top_movies.toPandas()
    if isinstance(movies_df, pd.DataFrame):
        movies_pandas_df = movies_df
    else:
        movies_pandas_df = movies_df.toPandas()

    merge_params = {"on": "movieId", "how": "left"}
    merge  = pd.merge(top_movies_df, movies_pandas_df, **merge_params)


    genre_dummies = merge['genres'].str.get_dummies(sep='|')
    genre_counts = genre_dummies.sum().sort_values(ascending=False)
    top_genres = genre_counts.head(10)
    return top_genres

def analyze_top_genres(split, movies_data):
    top_genres = top_movies_genres(split, movies_data)
    return top_genres

splits = [movies_largest_cluster_1, movies_largest_cluster_2, movies_largest_cluster_3]
split_percentages = [40, 60, 80]

for i, split in enumerate(splits):
    top_genres = analyze_top_genres(split, movies)
    print(f"Top Genres from High Rated Movies ({split_percentages[i]}% Split):")
    print(top_genres)

spark.stop()







