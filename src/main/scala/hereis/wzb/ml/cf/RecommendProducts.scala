package hereis.wzb.ml.cf

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

object RecommendProducts {
  def  main(args:Array[String]): Unit ={
    val conf = new SparkConf().setMaster("local").setAppName("recommendproducts")
    val sc = new SparkContext(conf)

    val data = sc.textFile("hdfs://192.168.1.120:9000/sample/data/test.data")
    val ratings = data.map(_.split(',') match{
      case Array(user,product,rate) =>
        Rating(user.toInt,product.toInt,rate.toDouble)
    })

    val rank = 2
    val numIterations = 2
    val model = ALS.train(ratings,rank,numIterations,0.01)
    val rs = model.recommendProducts(2,10)
    rs.foreach(println)

    sc.stop()
  }

}
