package common

import org.apache.spark.mllib.linalg

/**
 * Constructs used commonly throughout the application.
 */
object Common {
  case class LabeledSMSText(label: String, SMSText: String)
  case class LabeledAndCleanedSMSText(label: String, SMSText: String, cleanedSMSText: String)
  case class LabeledTokenizedSMSText(label: String, tokenizedSMSText: Array[String])
  case class LabeledTFVector(encodedLabel: Int, TFVector: linalg.Vector)

  case class DBInput(label: String, SMSText: String, cleanedSMSText: String, TFVector: Array[Double])

  final val CSVSeparator = '\t'
  final val path = "./src/main/resources/data/sms-spam-collection.tsv"

}
