package hereis.wzb.ml.linear

import scala.collection.mutable.HashMap

object SGD {
  var data = HashMap[Int,Int]()
  def getData():HashMap[Int,Int] = {
    for(i <- 1 to 50){
      data += ( i -> (12*i))
    }
    data
  }

  var Q:Double = 0      //第一步假设Q为0
  var a:Double = 0.1    //设置步进系数

  def sgd(x:Double,y:Double) = {
    Q = Q - a * ((Q * x) - y)    //设置迭代公式
  }

  def main(args: Array[String]): Unit ={
    val dataSource = getData()
    dataSource.foreach( myMap => {
      sgd(myMap._1,myMap._2)
    })
    println("最终结果 值为  "+ Q)
  }
}
