package exchange

import common.Common.MongoDBConf
import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.bson.BSONObject

object MongoDBWriter {
  val outputConfig: Configuration = new Configuration()
  outputConfig.set("mongo.output.format", outputFormatClass.getName)
  outputConfig.set("mongo.output.uri", MongoDBConf.outputURI)

  def write(documents: RDD[(Object, BSONObject)])(implicit sc: SparkContext) = {
    documents.saveAsNewAPIHadoopFile("file:///placeholder", keyClass, valueClass, outputFormatClass, outputConfig)
  }
}



