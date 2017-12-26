package hereis.wzb.ml.cf

import breeze.numerics.sqrt
import org.apache.spark.{SparkConf, SparkContext}

object CollaborativeFiltering {
  val conf = new SparkConf().setMaster("local").setAppName("collaborativefiltering")
  val sc = new SparkContext(conf)

  val users = sc.parallelize(Array("aaa","bbb","ccc","ddd","eee"))                                                   //设置用户
  val films = sc.parallelize(Array("亮剑","我是特种兵","两个小八路","高山下的花环","江湖恩仇录"))                   //设置影视剧名

  var source = Map[String,Map[String,Int]]()                                                               //使用一个source嵌套map作为姓名电影名和分值的存储
  val filmSource = Map[String,Int]()                                                                       //设置一个用以存放电影评分的map

  //设置电影评分
  def getSource():Map[String,Map[String,Int]] = {
    val user1FilmSource = Map("亮剑" -> 2,"我是特种兵"->3,"两个小八路"->1,"高山下的花环"->0,"江湖恩仇录"->1)
    val user2FilmSource = Map("亮剑" -> 1,"我是特种兵"->2,"两个小八路"->2,"高山下的花环"->1,"江湖恩仇录"->4)
    val user3FilmSource = Map("亮剑" -> 2,"我是特种兵"->1,"两个小八路"->0,"高山下的花环"->1,"江湖恩仇录"->4)
    val user4FilmSource = Map("亮剑" -> 3,"我是特种兵"->2,"两个小八路"->0,"高山下的花环"->5,"江湖恩仇录"->3)
    val user5FilmSource = Map("亮剑" -> 5,"我是特种兵"->3,"两个小八路"->1,"高山下的花环"->1,"江湖恩仇录"->2)
    source += ("aaa" -> user1FilmSource)  //对人名进行存储
    source += ("bbb" -> user2FilmSource)
    source += ("ccc" -> user3FilmSource)
    source += ("ddd" -> user4FilmSource)
    source += ("eee" -> user5FilmSource)
    source
  }

  //两两计算分值,采用余弦相似性 cos @ = ∑(x * y) / ( sqrt(∑x * x) * sqrt(∑y * y) )
  def getCollaborateSource(user1:String,user2:String):Double = {
    val user1FilmSource = source.get(user1).get.values.toVector     //获得第1个用户的评分
    val user2FilmSource = source.get(user2).get.values.toVector     //获得第2个用户的评分

    val member = user1FilmSource.zip(user2FilmSource).map(d => d._1 * d._2).reduce(_+_).toDouble  //对公式分子部分进行计算

    //求出分母第1个变量值
    val temp1 = math.sqrt(user1FilmSource.map(num => {
      math.pow(num,2)
    }).reduce(_+_))

    //求出分母第2个变量值
    val temp2 = math.sqrt(user2FilmSource.map(num => {
      math.pow(num,2)
    }).reduce(_+_))

    val denominator = temp1 * temp2                                     //求出分母
    member / denominator                                                 //进行计算
  }

  //两两计算分值,采用欧几里得相似性 d = sqrt(∑((x - y) * (x - y)) )
  def getCollaborateSource2(user1:String,user2:String):Double = {
    val user1FilmSource = source.get(user1).get.values.toVector     //获得第1个用户的评分
    val user2FilmSource = source.get(user2).get.values.toVector     //获得第2个用户的评分

    val member = user1FilmSource.zip(user2FilmSource).map(d => math.pow((d._1 - d._2),2)).reduce(_+_).toDouble
    sqrt(member)    //开方
  }


  def main(args:Array[String]): Unit ={
    getSource()                                                          //初始化分数
    val name="bbb"                                                      //设定目标对象
    users.foreach(user => {                                               //迭代进行计算
      println(name + "  相对于  "+user +" 的相似性分数是： "+getCollaborateSource2(name,user))
    })

  }

}
