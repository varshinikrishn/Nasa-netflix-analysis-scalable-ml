import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, dayofmonth, hour, max, regexp_extract, split, size
from pyspark.sql.functions import to_timestamp
from pyspark.sql import functions as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# create a spark session
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Log Analysis") \
    .config("spark.local.dir", os.environ['TMPDIR']) \
    .getOrCreate()

# create spark context
sc = spark.sparkContext

# set log level to warn
sc.setLogLevel("WARN")


# read the log file as text and cache it for efficient repeated access
log_file_path = "/users/acq22vk/com6012/ScalableML/Data/NASA_access_log_Jul95.gz"
log_file_df = spark.read.text(log_file_path).cache()

# regex pattern to extract the host from each log entry
pattern_host = r'^(\S+)'

# extract relevant columns using regular expressions and splitting methods
data = log_file_df.withColumn('host', regexp_extract('value', pattern_host, 1)) 
data = data.withColumn('timestamp', regexp_extract('value', r'.* - - \[(.*)\].*', 1)) 
data = data.withColumn('request', regexp_extract('value', r'.*\"(.*)\".*', 1)) 
data = data.withColumn('HTTP reply code', split('value', ' ').getItem(size(split('value', ' '))-2).cast("int")) 
data = data.withColumn('bytes in the reply', split('value', ' ').getItem(size(split('value', ' ')) - 1).cast("int")) 
data = data.drop("value")  

# remove rows with null values to clean the data
data = data.na.drop()

# cache the dataframe to optimize subsequent actions
data.cache()

print('-'*100) # task A

# count the number of requests from Germany by filtering hosts ending with '.de'
numRequestsGermany = data.filter(col('host').endswith('.de')).count()
# count the number of requests from Canada by filtering hosts ending with '.ca'
numRequestsCanada = data.filter(col('host').endswith('.ca')).count()
# count the number of requests from Singapore by filtering hosts ending with '.sg'
numRequestsSingapore = data.filter(col('host').endswith('.sg')).count()

# print the total number of requests for each country
print(f"Total number of requests from Germany: {numRequestsGermany}")
print(f"Total number of requests from Canada: {numRequestsCanada}")
print(f"Total number of requests from Singapore: {numRequestsSingapore}")

# plot the bar chart for total number of requests by country
countries = ['Germany', 'Canada', 'Singapore']
counts = [numRequestsGermany, numRequestsCanada, numRequestsSingapore]
plt.bar(countries, counts, color=[ 'cyan','gray', 'brown'])

# add title and label
plt.title('Number of Requests by Country')
plt.xlabel('Country')
plt.ylabel('Number of Requests')
plt.savefig('/users/acq22vk/com6012/ScalableML/Output/Q1_figA.jpg', dpi=200, bbox_inches="tight")
plt.close()

print('-'*100) # task B

# counts the number of unique hosts
def count_unique_hosts_by_country(data, country_code):
    
    pattern = f'.*\\.{country_code}$'
    
    return data.filter(F.col('host').rlike(pattern)).agg(F.countDistinct('host').alias('unique_hosts')).collect()[0]['unique_hosts']

# count unique hosts for Germany
GermanyUniqueHosts = count_unique_hosts_by_country(data, 'de')
# count unique hosts for Canada
CanadaUniqueHosts = count_unique_hosts_by_country(data, 'ca')
# count unique hosts for Singapore
SingaporeUniqueHosts = count_unique_hosts_by_country(data, 'sg')

print(f"The number of unique hosts Germany has is {GermanyUniqueHosts}")
print(f"The number of unique hosts Canada has is {CanadaUniqueHosts}")
print(f"The number of unique hosts Singapore has is {SingaporeUniqueHosts}")

# function to return tophost
def topHostsByCountry(data, country_code, limit=9):
    top_hosts = data.filter(data.host.endswith(country_code))\
        .groupBy('host')\
        .count()\
        .orderBy(desc('count'))\
        .limit(limit)\
        .collect()
    return top_hosts

GermanyTopHosts = topHostsByCountry(data, '.de')
CanadaTopHosts = topHostsByCountry(data, '.ca')
SingaporeTopHosts = topHostsByCountry(data, '.sg')

# function to display the top hosts for a given country

def top_hosts(countryHosts, countryName):
    print("-" * 100)
    print(f"The top 9 most frequent hosts in {countryName}:")
    # convert spark to pandas dataframe
    topHosts_df = countryHosts.limit(9).toPandas()
    pd.set_option('display.max_rows', None)
    print(topHosts_df.to_string(index=False))

# creating DataFrames for top hosts data of different countries 
GermanyTopHosts_df = spark.createDataFrame(GermanyTopHosts)
CanadaTopHosts_df = spark.createDataFrame(CanadaTopHosts)
SingaporeTopHosts_df = spark.createDataFrame(SingaporeTopHosts)

# display top hosts 
top_hosts(GermanyTopHosts_df, "Germany")
top_hosts(CanadaTopHosts_df, "Canada")
top_hosts(SingaporeTopHosts_df, "Singapore")


print('-'*100) # task C

# function to calculate percentages 
def percentages(CountryCode, countryName):

    # filtering data for hosts with the specified country code, grouping by host, and counting occurrences
    topHosts_df = data.filter(data.host.endswith(CountryCode)) \
                      .groupBy('host').count().orderBy(F.desc('count'))

    # calculating the count of the rest
    topHosts_counts = topHosts_df.limit(9)
    totalRequests = data.filter(data.host.endswith(CountryCode)).count()
    
    # sum of counts from the top hosts using Spark's aggregation
    total_topHosts_requests = topHosts_counts.groupBy().sum('count').collect()[0][0]
    remaining_requests = totalRequests - total_topHosts_requests
    
    # add rest' category with the remaining requests
    rest_df = spark.createDataFrame([('Rest', remaining_requests)], ['host', 'count'])
    final_df = topHosts_counts.union(rest_df)
    
    # collect data for plotting
    final_list = final_df.collect()
    hosts, counts = zip(*[(row['host'], row['count']) for row in final_list])
    percentages = [count / totalRequests * 100 for count in counts]
    
    # bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(hosts, percentages, color='blue', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10, color='black')
    
    plt.title(f'Percentage of Requests by Most Frequent Host for {countryName}', fontsize=14)
    plt.xlabel('Hosts', fontsize=12)
    plt.ylabel('Percentage of Total Requests', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(f'/users/acq22vk/com6012/ScalableML/Output/Q1_figC_{countryName}.jpg', dpi=300, bbox_inches='tight')
    plt.close()

percentages('.de', 'Germany')
percentages('.ca', 'Canada')
percentages('.sg', 'Singapore')


print('-'*100) # task D

countries = ['Germany', 'Canada', 'Singapore']

 # extracts the day and hour more efficiently
def extract_day_hour(timestamp):
 parts = timestamp.split(':')
 day = int(parts[0][-2:]) # Get last two characters for day
 hour = int(parts[1]) # Get hour directly
 return (hour, day)

 # initialize heatmap data array
def plot_heatmap(data, title):
    heatmap_data = np.zeros((24, 31))    
    for day, hour, count in data:
        heatmap_data[hour, day-1] = count  # Adjusted indices based on Python's zero-indexing

    # set the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel('Day')
    ax.set_ylabel('Hour')
    ax.set_xticks(np.arange(0.5, 31.5))  # Set ticks to be in the middle of the cells
    ax.set_yticks(np.arange(0.5, 24.5))  # Set ticks to be in the middle of the cells
    ax.set_xticklabels(np.arange(1, 32))
    ax.set_yticklabels(np.arange(0, 24))

    # create heatmap
    heatmap = ax.pcolor(heatmap_data, cmap=plt.cm.Greens, edgecolors='k', linewidths=0.5)
    colorbar = plt.colorbar(heatmap)
    colorbar.set_label('Number of Visits')

    plt.savefig('/users/acq22vk/com6012/ScalableML/Output/Q1_figD_{}.png'.format(title), dpi=200, bbox_inches="tight")
    plt.close()

# function to retrieve the top host for a given country domain
def get_top_host(country_domain):
    return data.filter(data.host.like(f"%{country_domain}")) \
               .groupBy('host').count() \
               .orderBy(desc('count')) \
               .limit(1) \
               .collect()[0].host

# top hosts for each country
GermanyTopHosts = get_top_host('.de')
CanadaTopHosts = get_top_host('.ca')
SingaporeTopHosts = get_top_host('.sg')

def prepare_heatmap_data(top_host):
    # Selecting timestamp where the host matches the top host
    host_data = data.select('timestamp').filter(data.host == top_host)
    
    # Extracting day and hour, converting to DataFrame
    host_data_transformed = host_data.rdd.map(lambda x: extract_day_hour(x.timestamp)).toDF(['hour', 'day'])
    
    # grouping by day and hour, counting visits, and sorting
    return host_data_transformed.groupBy(['day', 'hour']).count().orderBy('day', 'hour')

# preparing data for Germany, Canada, and Singapore
Germany_data = prepare_heatmap_data(GermanyTopHosts)
Canada_data = prepare_heatmap_data(CanadaTopHosts)
Singapore_data = prepare_heatmap_data(SingaporeTopHosts)

# function to prepare heatmap data
def prepare_and_plot_heatmap(country_data, country_name):
    filtered_data = country_data.filter(country_data.day <= 31)
    
    # collect data once and then pass it to the plotting function
    data_for_plotting = filtered_data.collect()
    plot_heatmap(data_for_plotting, country_name)

# heatmap generation
prepare_and_plot_heatmap(Germany_data, 'Germany')
prepare_and_plot_heatmap(Canada_data, 'Canada')
prepare_and_plot_heatmap(Singapore_data, 'Singapore')


spark.stop()








