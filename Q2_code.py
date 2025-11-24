import os
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StandardScaler, StringIndexer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# create a spark session
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Liability Prediction") \
    .config("spark.local.dir", os.environ['TMPDIR']) \
    .getOrCreate()

# create spark context
sc = spark.sparkContext

# set log level to warn
sc.setLogLevel("WARN")

# read the dataset from the csv file 
data = spark.read.csv('/users/acq22vk/com6012/ScalableML/Data/freMTPL2freq.csv', header=True, inferSchema=True)
# cache the data to optimize reading speed
data.cache()
# new column 'hasClaim' to indicate if there were any claims (1) or not (0)
data = data.withColumn("hasClaim", when(col("ClaimNb") > 0, 1).otherwise(0))

# fractions for stratified sampling to split data into training and test sets
fractions = {0: 0.7, 1: 0.3}  # 70% for no claims, 30% for claims
# sample the data by 'hasClaim' 
train = data.sampleBy("hasClaim", fractions, seed=24165)
# create a test set 
test = data.subtract(train)

# define feature categories
feature_columns = ['Exposure', 'Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Density', 'Region']
categorical_columns = []
for col in feature_columns:
    if data.schema[col].dataType == StringType():
        categorical_columns.append(col)
numeric_columns = []
for col in feature_columns:
    if col not in categorical_columns:
        numeric_columns.append(col)

# set the ML pipeline
indexers = []
for col in categorical_columns:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
    indexers.append(indexer)
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=f"{indexer.getOutputCol()}_ohe") for indexer in indexers]
assembler_inputs = [encoder.getOutputCol() for encoder in encoders] + numeric_columns
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# define models
glm_poisson_distribution = GeneralizedLinearRegression(family="poisson", link="log", labelCol="ClaimNb")
logreg_l1 = LogisticRegression()
logreg_l1.setLabelCol("hasClaim")
# set the elastic net parameter to 1 for L1 regularization
logreg_l1.setElasticNetParam(1)

logreg_l2 = LogisticRegression()
logreg_l2.setLabelCol("hasClaim")
logreg_l2.setElasticNetParam(0)

# separate pipelines for each model
pipeline_poisson = Pipeline(stages=indexers + encoders + [assembler, scaler, glm_poisson_distribution])
pipeline_logreg_l1 = Pipeline(stages=indexers + encoders + [assembler, scaler, logreg_l1])
pipeline_logreg_l2 = Pipeline(stages=indexers + encoders + [assembler, scaler, logreg_l2])

# set parameter grids for cross-validation
paramGrid_poisson = ParamGridBuilder().addGrid(glm_poisson_distribution.regParam, [0.001, 0.01, 0.1, 1, 10]).build()
paramGrid_lr = ParamGridBuilder().addGrid(logreg_l1.regParam, [0.001, 0.01, 0.1, 1, 10]).addGrid(logreg_l2.regParam, [0.001, 0.01, 0.1, 1, 10]).build()

# define the estimator, which is a pipeline configured for Poisson regression
estimator = pipeline_poisson
paramGrid = paramGrid_poisson
evaluator = RegressionEvaluator(labelCol="ClaimNb")

# initialize the CrossValidator
cross_validation_poisson = CrossValidator()
cross_validation_poisson.setEstimator(estimator)
cross_validation_poisson.setEstimatorParamMaps(paramGrid)

# set the evaluator
cross_validation_poisson.setEvaluator(evaluator)
# set the number of folds for cross-validation
cross_validation_poisson.setNumFolds(3)

# define the estimator as the logistic regression pipeline with L1 regularization
estimator_logreg_l1 = pipeline_logreg_l1
paramGrid_logreg_l1 = paramGrid_lr
evaluator_logreg_l1 = BinaryClassificationEvaluator(labelCol="hasClaim")

# initialize the CrossValidator for the logistic regression model
crossval_logreg_l1 = CrossValidator()
crossval_logreg_l1.setEstimator(estimator_logreg_l1)
crossval_logreg_l1.setEstimatorParamMaps(paramGrid_logreg_l1)
crossval_logreg_l1.setEvaluator(evaluator_logreg_l1)
crossval_logreg_l1.setNumFolds(3)

# define the estimator as the logistic regression pipeline with L2 regularization
estimator_logreg_l2 = pipeline_logreg_l2
paramGrid_logreg_l2 = paramGrid_lr
evaluator_logreg_l2 = BinaryClassificationEvaluator(labelCol="hasClaim")

# initialize the CrossValidator for the logistic regression model
crossval_logreg_l2 = CrossValidator()
crossval_logreg_l2.setEstimator(estimator_logreg_l2)
crossval_logreg_l2.setEstimatorParamMaps(paramGrid_logreg_l2)
crossval_logreg_l2.setEvaluator(evaluator_logreg_l2)
crossval_logreg_l2.setNumFolds(3)

# training models using a reduced sample for cross-validation
train_sampled = train.sample(False, 0.1, seed=24165)
poisson_crossval_model = cross_validation_poisson.fit(train_sampled)
l1_logreg_cv_model = crossval_logreg_l1.fit(train_sampled)
l2_logreg_cv_model = crossval_logreg_l2.fit(train_sampled)

# utilizing best models to predict
optimal_poisson_model = poisson_crossval_model.bestModel
poisson_model_predictions = optimal_poisson_model.transform(test)

optimal_logreg_l1_model = l1_logreg_cv_model.bestModel
predictions_logreg_l1 = optimal_logreg_l1_model.transform(test)

optimal_logreg_l2_model = l2_logreg_cv_model.bestModel
predictions_logreg_l2 = optimal_logreg_l2_model.transform(test)

# evaluating model performance
poisson_model_evaluator = RegressionEvaluator(labelCol="ClaimNb", metricName="rmse")
poisson_model_rmse = poisson_model_evaluator.evaluate(poisson_model_predictions)

# define the evaluator for binary classification
binary_evaluator = BinaryClassificationEvaluator(labelCol="hasClaim", metricName="areaUnderROC")

# evaluate the models
auc_logreg_l1 = binary_evaluator.evaluate(predictions_logreg_l1)
auc_logreg_l2 = binary_evaluator.evaluate(predictions_logreg_l2)

# Accuracy calculation for Logistic Regression models
accuracy_logreg_l1 = predictions_logreg_l1.withColumn('correct', expr("float(prediction = hasClaim)")).selectExpr("AVG(correct)").first()[0]
accuracy_logreg_l2 = predictions_logreg_l2.withColumn('correct', expr("float(prediction = hasClaim)")).selectExpr("AVG(correct)").first()[0]

print('-'*100)
# consolidate model evaluations and result printing
model_evaluations = {
    "Poisson Model RMSE": poisson_model_rmse,
    "AUC for Logistic Regression with L1 regularization ": auc_logreg_l1,
    "AUC for Logistic Regression with L2 regularization ": auc_logreg_l2,
    "Accuracy for Logistic Regression with L1 regularization ": accuracy_logreg_l1,
    "Accuracy for Logistic Regression with L2 regularization ": accuracy_logreg_l2
}

# print model performance metrics
for model_name, metric_value in model_evaluations.items():
    print(f" {model_name} : {metric_value}\n ")

# print coefficients of the models
print("Poisson model coefficients : \n ", optimal_poisson_model.stages[-1].coefficients)
print(" \n L1 regularized logistic regression model coefficients : \n ", optimal_logreg_l1_model.stages[-1].coefficients)
print(" \n L2 regularized logistic regression model coefficients : \n ", optimal_logreg_l2_model.stages[-1].coefficients)

print('-'*100)

spark.stop()

