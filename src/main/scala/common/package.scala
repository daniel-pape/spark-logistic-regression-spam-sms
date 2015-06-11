package common

import org.apache.spark.mllib.linalg

/**
 * Constructs used throughout the application.
 */
package object Common {
  case class LabeledSMSText(label: String, SMSText: String)
  case class LabeledTokenizedSMSText(label: String, tokenizedSMSText: Array[String])
  case class LabeledTFVector(encodedLabel: Int, TFVector: linalg.Vector)

  case class labeledData(label: String, SMSText: String, cleanedSMSText: String, TFVector: Array[Double]) {
    override def toString() = s"labeledData($label, $SMSText, $cleanedSMSText, ${TFVector.toList})"
  }

  object MongoDBConf {
    def server = "localhost"
    def port = 27017
    def dbName = "spam_sms_db"
    def collectionName = "spam_sms"
    def inputURI = s"mongodb://$server:$port/$dbName.$collectionName"
    def outputURI = s"mongodb://$server:$port/$dbName.${collectionName}1"
  }

  final val CSVSeparator = '\t'
  final val path = "./src/main/resources/data/sms-spam-collection.tsv"
}
