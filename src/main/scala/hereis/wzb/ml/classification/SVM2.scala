package hereis.wzb.ml.classification

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object SVM2 {
  val conf = new SparkConf().setMaster("local").setAppName("svm2")
  val sc = new SparkContext(conf)

  def main(args:Array[String]): Unit ={
    val data = MLUtils.loadLibSVMFile(sc,"hdfs://192.168.1.120:9000/sample/data/wa.txt")
    val splits = data.randomSplit(Array(0.7,0.3),seed = 11L)
    val parsedata = splits(0)
    val testdata = splits(1)

    val model = SVMWithSGD.train(parsedata,50)

    val predictionAndLabels = testdata.map{
      case LabeledPoint(label,features) =>
        val prediction = model.predict(features)
        (prediction,label)
    }

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    println("Presision = "+precision)

    val patient = Vectors.dense(Array(70,3,180.0,4,3))
    if(patient == 1)
      println("患者的胃癌有几率转移。")
    else
      println("患者的胃癌的没有几率转移。")

    sc.stop()
  }

}
