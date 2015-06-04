package smsClassificationWithLogRegr

import java.io.StringReader

import com.opencsv.CSVReader
import common.Common._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import preprocessing.{DocumentVectorizer, LineCleaner}

import scala.collection.Map

/**
 * Executable Spark driver programme. Use VM option "-Dspark.master=local" in the run
 * configuration as well as $MODULE_DIR$ as working directory.
 */
object SpamClassificationViaLogisticRegression extends App {
  def getTopWordsList(labeledTokenizedSmsTexts: RDD[LabeledTokenizedSMSText]): List[String] = {
    val words = labeledTokenizedSmsTexts.flatMap(lbldText => lbldText.tokenizedSMSText)
    val wordCount = words.map(word => (word, 1)).reduceByKey((k, l) => k + l).collect().size
    val wordFrequencyMap: Map[String, Long] = words.countByValue()

    val threshold = 136
    val topWords = wordFrequencyMap.filter { case (word, freq) => freq >= threshold}
    val topWordsList: List[String] = topWords.keySet.toList

    topWordsList
  }

  val sc = new SparkContext(new SparkConf().setAppName("SpamClassificationViaLogisticRegression"))
  val input: RDD[String] = sc.textFile(path)

  /**
   * Contains the [[common.Common.LabeledSMSText]]s that are correctly parsed from the input file.
   */
  val labeledSmsTexts: RDD[LabeledSMSText] = input.map { line =>
    val reader = new CSVReader(new StringReader(line), CSVSeparator)

    try {
      val nextLine: Option[List[String]] = Option(reader.readNext()).map(_.toList)

      nextLine match {
        case Some(line) if line.length == 2 =>
          val label = line(0)
          val smsText = line(1)
          val smsTextCleaned = LineCleaner.clean(smsText)
          val result = LabeledSMSText(label, smsTextCleaned)

          Some(result)
        case Some(line) if line.length != 2 =>
          // The `line` was not parsed correctly and we ignore subsequently.
          Option.empty[LabeledSMSText]
        case None =>
          Option.empty[LabeledSMSText]
      }
    } catch {
      case ex: java.io.IOException =>
        val msg = s"Exception while reading $path. Message:\n${ex.getMessage}"
        println(msg)
        Option.empty[LabeledSMSText]
    }
  }.filter(_.isDefined).map(_.get)

  /**
   * Contains the tokenized SMS texts together with their labels.
   * @note We cache this RDD because it will be re-used subsequently.
   */
  val labeledTokenizedSmsTexts: RDD[LabeledTokenizedSMSText] = labeledSmsTexts.map {
    labeledSMSText => LabeledTokenizedSMSText(labeledSMSText.label, labeledSMSText.SMSText.split(" "))
  }.cache()

  /**
   * The list of words used to build the term frequency vectors.
   */
  val wordList = getTopWordsList(labeledTokenizedSmsTexts)

  /**
   * Contains the labeled TF vectors where the label for "ham" is encoded as 0 and the
   * one for "spam" as 1.
   */
  val labeledTFVectors: RDD[LabeledTFVector] = labeledTokenizedSmsTexts.map {
    labeledTokenizedSmsText =>
      val encodedLabel = if (labeledTokenizedSmsText.label == "ham") 0 else 1
      val TFVector = DocumentVectorizer.vectorize(labeledTokenizedSmsText.tokenizedSMSText, wordList)
      LabeledTFVector(encodedLabel, TFVector)
  }

  /**
   * Contains the data for building the logistic regression model.
   */
  val data: RDD[LabeledPoint] = labeledTFVectors.map {
    labeledTfVector =>
      LabeledPoint(labeledTfVector.encodedLabel, labeledTfVector.TFVector)
  }

  val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
  val training = splits(0).cache()
  val test = splits(1)

  val logRegHelper = LogisticRegressionHelper(training, test)

  logRegHelper.performLogReg()
  logRegHelper.performLogRegWithWeightsGiven()
}
