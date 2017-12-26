package hereis.wzb.ml.clustering

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

object kmean {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("kmean").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("hdfs://192.168.1.120:9000/sample/data/kmean.txt")
    val parsedata = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))
      .cache()

    val numclusters = 2
    val numIterations = 20
    val model = KMeans.train(parsedata,numclusters,numIterations)

    model.clusterCenters.foreach(println)
    sc.stop()
  }

}
