// Databricks notebook source
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.ml.evaluation.RegressionEvaluator

// COMMAND ----------

val rawData = sc.textFile("dbfs:/FileStore/shared_uploads/tamaraychowdhury@gmail.com/u.data")
rawData.first()

// COMMAND ----------

case class Rating(userId: Int, movieId: Int, rating: Float, timeStamp: Long)
def parseString(str: String): Rating = {
    val fields = str.split("\t")
    assert(fields.size==4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
}

// COMMAND ----------

val ratings = rawData.map(parseString).toDF()
ratings.show(3)

// COMMAND ----------

val Array(train, test) = ratings.randomSplit(Array(0.8, 0.2))
println(train.count, rawData.count)

// COMMAND ----------

train.show(3)

// COMMAND ----------

val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

// COMMAND ----------

val model = als.fit(train)

// COMMAND ----------

model.setColdStartStrategy("drop")
val predictions = model.transform(test)
predictions.show(3)

// COMMAND ----------

val eval = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
val rmse = eval.evaluate(predictions)

// COMMAND ----------


