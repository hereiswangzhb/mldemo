package hereis.wzb.ml.clustering

import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

object gmg {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("gmg")
    val sc = new SparkContext(conf)

    val data = sc.textFile("hdfs://192.168.1.120:9000/sample/data/gmg.txt")
    val parsedata = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

    val model = new GaussianMixture().setK(2).run(parsedata)
    for(i <- 0 until model.k){
      println("weight = %f\nmu = %s\nsigma=\n%s\n" format(model.weights(i),model.gaussians(i).mu,model.gaussians(i).sigma))
    }
  }


}
