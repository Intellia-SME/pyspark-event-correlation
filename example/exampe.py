from pyspark.sql import SparkSession

from pyspark_event_correlation.classifier import EventCorrelationClassifier

spark = SparkSession.builder.appName('data_processing').getOrCreate() 

training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

model = EventCorrelationClassifier(maxIter=10, regParam=0.3, elasticNetParam=0.8)


fit_model = model.fit(training)

print("Coefficients: " + str(fit_model.coefficients))
print("Intercept: " + str(fit_model.intercept))