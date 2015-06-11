package exchange

import common.Common.MongoDBConf
import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.bson.BSONObject

object MongoDBReader {
  val inputConfig: Configuration = new Configuration()
  inputConfig.set("mongo.job.input.format", inputFormatClass.getName)
  inputConfig.set("mongo.input.uri", MongoDBConf.inputURI)

  def read(implicit sc: SparkContext): RDD[(Object, BSONObject)] = {
    val documents = sc.newAPIHadoopRDD(inputConfig, inputFormatClass, keyClass, valueClass)
    documents
  }
}
