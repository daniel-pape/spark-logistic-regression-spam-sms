package preprocessing

import java.io.StringReader

import com.opencsv.CSVReader
import common.Common._

import scala.collection.Map
import scala.io.Source._

object CreateWordList {
  def createWordFrequencyMap(words: List[String]) = {
    val initialWordFrequencyMap = scala.collection.mutable.Map[String, Int]()

    val wordFrequencyMap: Map[String, Int] = words.foldLeft(initialWordFrequencyMap) {
      (acc, word) =>
        if (acc.contains(word)) {
          acc(word) += 1
        } else {
          acc(word) = 1
        }

        acc
    }

    wordFrequencyMap
  }

  def getTopWordsList(labeledTokenizedSmsTexts: List[LabeledTokenizedSMSText]): List[String] = {
    val words = labeledTokenizedSmsTexts.flatMap(lbldText => lbldText.tokenizedSMSText)
    val wordFrequencyMap = createWordFrequencyMap(words)
    val threshold = 136
    val topWords = wordFrequencyMap.filter { case (word, freq) => freq >= threshold}
    val topWordsList: List[String] = topWords.keySet.toList

    topWordsList
  }

  val input = fromFile(path).getLines().toList

  /**
   * Contains the [[common.Common.LabeledSMSText]]s that are correctly parsed from the input file.
   */
  val labeledSmsTexts: List[LabeledSMSText] = input.map { line =>
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
  val labeledTokenizedSmsTexts: List[LabeledTokenizedSMSText] = labeledSmsTexts.map {
    labeledSMSText => LabeledTokenizedSMSText(labeledSMSText.label, labeledSMSText.SMSText.split(" "))
  }

  /**
   * The list of words used to build the term frequency vectors.
   */
  val wordList = getTopWordsList(labeledTokenizedSmsTexts)
}
