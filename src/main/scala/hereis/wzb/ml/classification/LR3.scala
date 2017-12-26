package hereis.wzb.ml.classification

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object LR3 {
  val conf = new SparkConf().setMaster("local").setAppName("lr3")
  val sc = new SparkContext(conf)

  def main(args:Array[String]): Unit ={
    val data = MLUtils.loadLibSVMFile(sc,"hdfs://192.168.1.120:9000/sample/data/sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.6,0.4),seed = 11L)
    val parsedData = splits(0)
    val testData = splits(1)
    val model = LogisticRegressionWithSGD.train(parsedData,50)
    println(model.weights)
    val predictionAndLabels = testData.map{
      case LabeledPoint(label,features) =>
        val prediction = model.predict(features)
        (prediction,label)
    }
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Presicion = "+ precision)

    sc.stop()
  }



}
