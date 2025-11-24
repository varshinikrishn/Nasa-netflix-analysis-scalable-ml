
import os
import re
import numpy as np
import warnings
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.param import Param

# create a spark session
spark = SparkSession.builder \
    .master("local[10]") \
    .appName("	Searching for exotic particles") \
    .config("spark.driver.memory", "30g") \
    .config("spark.local.dir", os.environ['TMPDIR']) \
    .getOrCreate()

# create spark context
sc = spark.sparkContext
# set log level to error
sc.setLogLevel("ERROR")
# ignore all warnings
warnings.filterwarnings("ignore")

# function to retrieve parameter 
def fetchParameter(hyper_params):
    hyper_params_list = []
    for param, value in hyper_params.items():
        param_name = param.name 
        param_parent = param.parent.split('_')[0]          
        hyper_params_list.append(f"{param_parent}.{param_name} = {value}")
    return hyper_params_list

print('-'*100)

# load data
data = spark.read.csv('/users/acq22vk/com6012/ScalableML/Data/HIGGS.csv', header=True, inferSchema=True)
features = ['label','lepton_pT','lepton_eta','lepton_phi', 'missing_energy_magnitude','missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_btag', 'jet_3_pt', 'jet_3_eta','jet_3_phi', 'jet_3_btag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_btag', 'mjj', 'mjjj', 'mlv', 'mjlv', 'mbb', 'mwbb', 'mwwbb']
n_col = len(data.columns)
database_schema_names = data.schema.names  
for i in range(n_col):
    data = data.withColumnRenamed(database_schema_names[i], features[i])      
string_columns = []
for x in data.schema.fields:
    if x.dataType == StringType():
        string_columns.append(x.name)
# Iterate over each column name in the list of string columns
for c in string_columns:
    # Cast the column to double type and update the DataFrame
    data = data.withColumn(c, col(c).cast("double"))

# filter the dataframe 
filtered_data = data.filter(data.label == 1)
pos_count = filtered_data.count()
filtered_data = data.filter(data.label == 0)
neg_count = filtered_data.count()

# Calculate the total number of rows in the DataFrame
total_rows = data.count()
minimum_count = min(pos_count, neg_count)
minor_fraction = minimum_count / float(total_rows)

# Calculate the total number of rows in the DataFrame
total_count = float(data.count())
minimum_count = min(pos_count, neg_count)
minor_fraction = minimum_count / total_count

# Initialize an empty dictionary for weights by class
weights_by_class = {}
weights_by_class[0] = minor_fraction
weights_by_class[1] = minor_fraction

# Convert the positive count to a floating point number for division
pos_count_float = float(pos_count)
bal_ratio = neg_count / pos_count_float

# split sampled data into training (70%) and testing (30%) subsets
data_sampled = data.sample(False, 0.01, seed=14).cache()
(sample_train_subset, sample_test_subset) = data_sampled.randomSplit([0.7, 0.3], seed=14)

# write to parquet
sample_train_subset.write.mode("overwrite").parquet('/users/acq22vk/com6012/ScalableML/Data/Q1training_subset.parquet')
sample_test_subset.write.mode("overwrite").parquet('/users/acq22vk/com6012/ScalableML/Data/Q1testing_subset.parquet')

# load from parquet file
training_subset = spark.read.parquet('/users/acq22vk/com6012/ScalableML/Data/Q1training_subset.parquet')
testing_subset = spark.read.parquet('/users/acq22vk/com6012/ScalableML/Data/Q1testing_subset.parquet')
print('-'*100)

# create a VectorAssembler with specified input and output columns
assembler = VectorAssembler(inputCols = features[1:], outputCol ="features") 

# define column name
label_col = "label"
feature_col = "features"
# maximum depth for the random forest
max_depth = 10
impurity = 'entropy'

# create a RandomForestClassifier with specified parameters
random_forest = RandomForestClassifier(labelCol=label_col,
                                       featuresCol=feature_col,
                                       maxDepth=max_depth,
                                       impurity=impurity)


# defining a pipeline for a random forest model 
random_forest_stages = [assembler, random_forest]
random_forest_pipeline = Pipeline(stages=random_forest_stages)
random_forest_paramGrid = ParamGridBuilder() \
    .addGrid(random_forest.maxDepth, [10, 5, 1]) \
    .addGrid(random_forest.maxBins, [10, 20, 50]) \
    .addGrid(random_forest.numTrees, [10, 15, 20]) \
    .build()

# defining cross validation for a random forest model 
random_forest_cross_validation = CrossValidator(estimator=random_forest_pipeline,
                          estimatorParamMaps=random_forest_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)
random_forest_cvModel = random_forest_cross_validation.fit(training_subset)
random_forest_predictions = random_forest_cvModel.transform(testing_subset)

# defining accuracy evaluator for random forest model 
accuracy_evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")          

# defining auc evaluator for a random forest model 
area_evaluator   = BinaryClassificationEvaluator\
      (labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# print accuracy and auc for random forest model 
random_forest_accuracy = accuracy_evaluator.evaluate(random_forest_predictions)
print( "Accuracy of sample Random Forest: %g " % random_forest_accuracy)
random_forest_area = area_evaluator.evaluate(random_forest_predictions)
print( "AUC for sample Random Forest: %g " % random_forest_area)
print('-'*100)

# pipeline for gradient boosting model 
gradient_boosting = GBTClassifier(maxIter=5, maxDepth=10 , labelCol="label", seed=14,
    featuresCol="features", lossType='logistic')
gradient_boosting_stages = [assembler, gradient_boosting]
gradient_boosting_pipeline = Pipeline(stages=gradient_boosting_stages)
gradient_boosting_paramGrid = ParamGridBuilder() \
    .addGrid(gradient_boosting.maxDepth, [5, 3, 10]) \
    .addGrid(gradient_boosting.maxIter, [30, 20, 10]) \
    .addGrid(gradient_boosting.stepSize, [0.2, 0.3, 0.05]) \
    .build()

# defining cross validation for a gradient boosting model 
gradient_boosting_crossvalidation = CrossValidator(estimator=gradient_boosting_pipeline,
                          estimatorParamMaps=gradient_boosting_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)
gradient_boosting_cvModel =gradient_boosting_crossvalidation.fit(training_subset)
gradient_boosting_predictions = gradient_boosting_cvModel.transform(testing_subset)

# defining accuracy for gradient boosting model 
gradient_boosting_accuracy = accuracy_evaluator.evaluate(gradient_boosting_predictions)
print( "Accuracy of sample Gradient Boost model = %g  "% gradient_boosting_accuracy)

# defining auc for gradient boosting model 
gradient_boosting_area = area_evaluator.evaluate(gradient_boosting_predictions)

# print accuracy and auc for gradient boosting model 
print( "AUC for sample Gradient Boost = %g  " % gradient_boosting_area)
print('-'*100)

# set neural network
layers = [len(features) - 1, 5, 4, 2]
neural_network = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=24165
)

neural_network_stages = [assembler, neural_network]
# pipeline for neural network model 
neural_network_pipeline = Pipeline(stages=neural_network_stages)

# parameters for neural network model 
neural_network_paramGrid = ParamGridBuilder() \
    .addGrid(neural_network.maxIter, [100, 10]) \
    .addGrid(neural_network.blockSize, [20, 30]) \
    .addGrid(neural_network.layers, [[len(features) - 1, 5, 4, 2], [len(features) - 1, 6, 5, 2]]) \
    .build()

# crossvalidation for neural network model 
neural_network_crossvalidation = CrossValidator(estimator=neural_network_pipeline,
                                    estimatorParamMaps=neural_network_paramGrid,
                                    evaluator=MulticlassClassificationEvaluator(),
                                    numFolds=5)
neural_network_cvModel = neural_network_crossvalidation.fit(training_subset)
neural_network_predictions = neural_network_cvModel.transform(testing_subset)

# accuracy for neural network model 
neural_network_accuracy = accuracy_evaluator.evaluate(neural_network_predictions)
# auc for neural network model 
neural_network_area = area_evaluator.evaluate(neural_network_predictions)

# print accuracy and auc for neural network model 
print( "Accuracy of the sample Neural Network = %g  " % neural_network_accuracy)
print( "AUC for sample Neural Network = %g  " % neural_network_area)

print('-'*100)
print('-'*100)
print('-'*100)


# extract the best hyperparameters for the Random Forest model      
random_forest_hyper = random_forest_cvModel.getEstimatorParamMaps()[np.argmax(random_forest_cvModel.avgMetrics)]
random_forest_params = fetchParameter(random_forest_hyper)   
print( "Random Forest Best Hyper-Parameters ")  
print(random_forest_params) 
print('-'*100)

# extract the best hyperparameters for the Gradient Boosted Tree model
gradient_boosting_hyper = gradient_boosting_cvModel.getEstimatorParamMaps()[np.argmax(gradient_boosting_cvModel.avgMetrics)]
gradient_boosting_params = fetchParameter(gradient_boosting_hyper)     
print( "Gradient Boosting Best Hyper-Parameters ") 
print(gradient_boosting_params)
print('-'*100)

# extract  the best hyperparameters for the Neural Network model
neural_network_hyper = neural_network_cvModel.getEstimatorParamMaps()[np.argmax(neural_network_cvModel.avgMetrics)]
neural_network_params = fetchParameter(neural_network_hyper) 
print( "Neural Network Best Hyper-Parameters ")  
print(neural_network_params) 
print('-'*100)
print('-'*100)


# read data from disk again to get fresh dataframes 
train = spark.read.parquet('/users/acq22vk/com6012/ScalableML/Data/Q1training_subset.parquet')
test = spark.read.parquet('/users/acq22vk/com6012/ScalableML/Data/Q1testing_subset.parquet')

# Random Forest
random_forest_best = random_forest_cvModel.bestModel
random_forest_predictions = random_forest_best.transform(test)

# evaluate Random Forest predictions
random_forest_accuracy = accuracy_evaluator.evaluate(random_forest_predictions)
random_forest_area = area_evaluator.evaluate(random_forest_predictions)
print( "Random Forest Accuracy = %g  " % random_forest_accuracy)
print( "Random Forest AUC = %g  " % random_forest_area)
print('-'*100)

# Gradient Boosting Trees
gradient_boosting_best = gradient_boosting_cvModel.bestModel
gradient_boosting_predictions = gradient_boosting_best.transform(test)

# evaluate gradient_boosting predictions
gradient_boosting_accuracy = accuracy_evaluator.evaluate(gradient_boosting_predictions)
gradient_boosting_area = area_evaluator.evaluate(gradient_boosting_predictions)
print( "Gradient Boosting Accuracy= %g  " % gradient_boosting_accuracy)
print( "Gradient Boosting AUC= %g  " % gradient_boosting_area)
print('-'*100)

# Neural Network
neural_network_best = neural_network_cvModel.bestModel
neural_network_predictions = neural_network_best.transform(test)

# evaluate neural_network predictions
neural_network_accuracy = accuracy_evaluator.evaluate(neural_network_predictions)
neural_network_area = area_evaluator.evaluate(neural_network_predictions)
print( "Neural Network Accuracy = %g  " % neural_network_accuracy)
print("Neural Network AUC = %g  " % neural_network_area)
print('-'*100)




