package hereis.wzb.ml.classification

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object LR {
  val conf = new SparkConf().setMaster("local").setAppName("lr")
  val sc = new SparkContext(conf)

  def main(args:Array[String]): Unit = {
    val data = sc.textFile("hdfs://192.168.1.120:9000/sample/data/u.txt")
    val parseddata = data.map{ line =>
      val parts = line.split('|')
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).toDouble))
    }.cache()

    val model = LogisticRegressionWithSGD.train(parseddata,50)
    val target = Vectors.dense(-1)
    val result = model.predict(target)
    println(result)

    sc.stop()
  }



}
