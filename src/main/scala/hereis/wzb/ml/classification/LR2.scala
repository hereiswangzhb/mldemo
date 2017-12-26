package hereis.wzb.ml.classification

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object LR2 {
  val conf = new SparkConf().setMaster("local").setAppName("lr2")
  val sc = new SparkContext(conf)

  def main(args : Array[String]): Unit ={
    val data = MLUtils.loadLibSVMFile(sc,"hdfs://192.168.1.120:9000/sample/data/sample_libsvm_data.txt")
    val model = LogisticRegressionWithSGD.train(data,50)
    println(model.weights.size)
    println(model.weights)
    println(model.weights.toArray.filter(_ != 0).size)

    sc.stop()
  }



}
