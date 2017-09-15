package com.ml.forestcover

import java.util

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

/**
  * Created by okorolenko on 10/06/2017.
  */
class ForestCoverPredictor(private val spark: SparkSession) {

  import spark.implicits._

}
trait Types

object ForestCoverPredictor {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("ForestCoverPredictor")
      .master("local[2]")
      .getOrCreate()

    import spark.implicits._

    val rawData = spark.read.textFile("src/main/resources/covdata/covtype.data").cache()
    val data = rawData.map(line => {
      val values = line.split(',').map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    })

    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()


    def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
      val predictionsAndLabels = data.map(example => (model.predict(example.features), example.label))
      new MulticlassMetrics(predictionsAndLabels)
    }

    val model = DecisionTree.trainClassifier(trainData.toJavaRDD, 7, Map[Int, Int](), "gini", 4, 100)
    val metrics = getMetrics(model, cvData.toJavaRDD)
    println(metrics.confusionMatrix)
    (0 until 7).map(cat => (metrics.precision(cat), metrics.recall(cat))).foreach(println(_))

    def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
      val countsByCategory = data.map(_.label).countByValue()
      val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
      counts.map(_.toDouble / counts.sum)
    }

    val trainPriorProbabilities = classProbabilities(trainData.toJavaRDD)
    val cvPriorProbabilities = classProbabilities(cvData.toJavaRDD)
    val finalProb = trainPriorProbabilities.zip(cvPriorProbabilities).map {
      case (trainProb, cvProb) => {
        println(s"trainProb=$trainProb cvProb=$cvProb")
        trainProb * cvProb
      }
    }.sum
    println(s"final prob=$finalProb")
  }


}