package common

import org.apache.spark.mllib.linalg

/**
 * Constructs used commonly throughout the application.
 */
package object Common {
  case class LabeledSMSText(label: String, SMSText: String)
  case class LabeledTokenizedSMSText(label: String, tokenizedSMSText: Array[String])
  case class LabeledTFVector(encodedLabel: Int, TFVector: linalg.Vector)

  case class labeledData(label: String, SMSText: String, cleanedSMSText: String, TFVector: Array[Double]) {
    override def toString() = s"labeledData($label, $SMSText, $cleanedSMSText, ${TFVector.toList})"
  }

  final val CSVSeparator = '\t'
  final val path = "./src/main/resources/data/sms-spam-collection.tsv"
}
