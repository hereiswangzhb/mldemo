package hereis.wzb.ml.linear

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

object LR {
  val conf = new SparkConf().setAppName("LR").setMaster("local")
  val sc = new SparkContext(conf)

  def main(args :Array[String]): Unit ={
    val data = sc.textFile("hdfs://192.168.1.120:9000/sample/data/lr.txt")
    val parsedata = data.map{
      line =>
        val parts = line.split('|')
        LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(',').map(_.toDouble)))
    }.cache()

    val model = LinearRegressionWithSGD.train(parsedata,2,0.1)
    val result = model.predict(Vectors.dense(2))

    println(result)

    sc.stop()
  }

}
