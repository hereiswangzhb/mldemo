package hereis.wzb.ml.classification

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object Bayes {
  val conf = new SparkConf().setMaster("local").setAppName("lr")
  val sc = new SparkContext(conf)

  def main(args:Array[String]): Unit = {
    val data = MLUtils.loadLibSVMFile(sc, "hdfs://192.168.1.120:9000/sample/data/sample_libsvm_data.txt")
    val model = NaiveBayes.train(data,1.0)
    model.labels.foreach(println)
    model.pi.foreach(println)

  }
}
