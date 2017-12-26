package hereis.wzb.ml.classification

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object SVM {
  val conf = new SparkConf().setMaster("local").setAppName("svm")
  val sc = new SparkContext(conf)

  def main(args:Array[String]): Unit ={
    val data = sc.textFile("hdfs://192.168.1.120:9000/sample/data/u.txt")
    val parsedata = data.map{ line =>
      val parts = line.split('|')
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).toDouble))
    }.cache()
    val model = SVMWithSGD.train(parsedata,10)
    println(model.weights)
    println(model.intercept)

    sc.stop()
  }

}
