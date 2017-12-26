package hereis.wzb.ml.classification

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object Bayes2 {

  def main(args:Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("lr")
    val sc = new SparkContext(conf)

    val data = sc.textFile("hdfs://192.168.1.120:9000/sample/data/nb.txt")
    val parsedata = data.map{ line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    val splits = data.randomSplit(Array(0.7,0.3),seed = 11L)
    val traindata = splits(0)
    val testdata = splits(1)
    val model = NaiveBayes.train(parsedata,lambda = 1.0)
//    val predictionAndLabel = testdata.map( p => (model.predict(p.features),p.label) )
//    val accuracy = 1.0 * predictionAndLabel.filter(
//      label => label._1 == label._2).count()
//    println(accuracy)


  }

}
