package dataExchange

import org.apache.spark.{SparkConf, SparkContext}

// Only for test purposes:
object TestDriver extends App {
  implicit val sc = new SparkContext(new SparkConf().setAppName("TestDriver"))
  val docs = MongoDBReader.read
  MongoDBWriter.write(docs)
}
