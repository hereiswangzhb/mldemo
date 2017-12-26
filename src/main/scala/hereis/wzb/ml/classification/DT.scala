package hereis.wzb.ml.classification

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object DT {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("dt").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"hdfs://192.168.1.120:9000/sample/data/tree.txt")
    val numclasses = 2
    val categoricalFeaturesInfo = Map[Int,Int]()
    val impurity = "entropy"
    val maxdepth = 5
    val maxbins = 3
    val model = DecisionTree.trainClassifier(data,numclasses,categoricalFeaturesInfo,impurity,maxdepth,maxbins)
    println(model.topNode)
    sc.stop()
  }

}
