package smsClassificationWithLogRegr

import java.io.StringReader
import java.util.regex._

import com.opencsv.CSVReader
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map
import scala.collection.immutable.ListMap

/**
 * Helper object that provides methods to clean-up lines of SMS text or normalize these
 * lines, e.g., replace typical patterns like number, urls, email addresses, etc. by
 * placeholder strings.
 *
 * @example
 * {{{
 *    import smsClassificationWithLogRegr.LineCleaner
 *    LineCleaner.normalizeCurrencySymbol("Replaces € in this text by \" normalizedcurrencysymbol \"")
 * }}}
 */
object LineCleaner {
  type SMSText = String

  /**
   * Returns `text` with all sub-strings matching the regular expression `regex`
   * replaced by the string `normalizationString`.
   *
   * @note This is used as common template for string normalization methods provided
   *       by this object.
   */
  def applyNormalizationTemplate(text: SMSText, regex: String, normalizationString: String): String = {
    val pattern = Pattern.compile(regex)
    val matcher = pattern.matcher(text)
    val normalizedText = matcher.replaceAll(normalizationString)

    normalizedText
  }

  /**
   * Returns `text` with the following occurrences removed:
   * 1) Punctuation: '.', ',', ':', '-', '!', '?' and combinations/repetitions of characters from 1) like '--',
   * '!?!?!', '...' (ellipses), etc.
   * 2) Special characters: '\n', '\t', '%', '#', '*', '|', '=', '(', ')', '"', '>', '<', '/'
   *
   * @note Use this with care if you are interested in phone numbers (they contain '-') or
   *       smileys (like ':-)').
   */
  def removePunctuationAndSpecialChars(text: SMSText): SMSText = {
    val regex = "[\\.\\,\\:\\-\\!\\?\\n\\t,\\%\\#\\*\\|\\=\\(\\)\\\"\\>\\<\\/]"
    val pattern = Pattern.compile(regex)
    val matcher = pattern.matcher(text)

    // Remove all matches, split at whitespace (allow repetitions) then join again.
    val cleanedText = matcher.replaceAll(" ").split("[ ]+").mkString(" ")

    cleanedText
  }

  /**
   * Returns `text` with every occurrence of one of the currency symbol
   * '$', '€' or '£' replaced by the string literal " normalizedcurrencysymbol ".
   */
  def normalizeCurrencySymbol(text: SMSText): SMSText = {
    val regex = "[\\$\\€\\£]"
    applyNormalizationTemplate(text, regex, " normalizedcurrencysymbol ")
  }

  /**
   * Returns `text` with every occurrence of an emonicon (see implementation for
   * details) replaced by the string literal " normalizedemonicon ".
   */
  def normalizeEmonicon(text: SMSText): SMSText = {
    val emonicons = List(":-)", ":)", ":D", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)")
    val regex = "(" + emonicons.map(Pattern.quote).mkString("|") + ")"
    applyNormalizationTemplate(text, regex, " normalizedemonicon ")
  }

  /**
   * Returns `text` with every occurrence of one of a number
   * replaced by the string literal "normalizednumber".
   */
  def normalizeNumbers(text: SMSText): SMSText = {
    val regex = "\\d+"
    applyNormalizationTemplate(text, regex, " normalizednumber ")
  }

  /**
   * Returns `text` with every occurrence of one of an URL
   * replaced by the string literal " normalizedurl ".
   *
   * @note This implementation does only a very naive test and
   *       also will miss certain cases.
   */
  def normalizeURL(text: SMSText): SMSText = {
    val regex = "(http://|https://)?www\\.\\w+?\\.(de|com|co.uk)"
    applyNormalizationTemplate(text, regex, " normalizedurl ")
  }

  /**
   * Returns `text` with every occurrence of one of an email address
   * replaced by the string literal " normalizedemailadress ".
   *
   * @note This implementation does only a very naive test and
   *       also will miss certain cases.
   */
  def normalizeEmailAddress(text: SMSText): SMSText = {
    val regex = "\\w+(\\.|-)*\\w+@.*\\.(com|de|uk)"
    applyNormalizationTemplate(text, regex, " normalizedemailadress ")
  }

  /**
   * Returns `line` with HTML character entities, excluding whitespace "&nbsp;"
   * which will be treated elsewhere, removed.
   */
  def removeHTMLCharacterEntities(text: SMSText): SMSText = {
    val HTMLCharacterEntities = List("&lt;", "&gt;", "&amp;", "&cent;", "&pound;", "&yen;", "&euro;", "&copy;", "&reg;")
    val regex = "(" + HTMLCharacterEntities.map(x => "\\" + x).mkString("|") + ")"
    val pattern = Pattern.compile(regex)
    val matcher = pattern.matcher(text)

    val cleanedText = matcher.replaceAll("")

    cleanedText
  }

  /**
   * First normalizes the `text` and then removes unwanted characters from it.
   */
  def clean(text: SMSText): SMSText = {
    List(text).map(text => text.toLowerCase())
      .map(normalizeEmonicon)
      .map(normalizeURL)
      .map(normalizeEmailAddress)
      .map(normalizeCurrencySymbol)
      .map(removeHTMLCharacterEntities)
      .map(normalizeNumbers)
      .map(removePunctuationAndSpecialChars)
      .head
  }
}

/**
 * Helper object that provides the `vectorize` method to produce
 * a term frequency vector based for an input document based on list of
 * words that are used as index.
 *
 * @example The following code
 *          {{{
 *             import smsClassificationWithLogRegr.DocumentVectorizer
 *             val input = "To be or not to be that is the question" + "That is utter rubbish"
 *             val document = input.split(" ")
 *             val wordList = List("be", "not", "To", "unused")
 *             DocumentVectorizer.vectorize(document, wordList)
 *          }}}
 *          would produce the term frequency vector `Vectors.dense(Array(2.0, 1.0, 2.0, 0.0))`.
 */
object DocumentVectorizer {
  /**
   * Returns the term frequency vector for the document `document` based
   * on the reference index `wordList`. Note: All words will be treated as lower case
   * and the word list will be used in its alphabetically sorted version to build the
   * entries of the term frequency vector.
   *
   * @param document Array containing the document tokenized to [[String]]s.
   * @param wordList List of words which in alphabetical order serve as associative indices for the vector.
   * @return The term frequency vector obtained from the document based on the lift of words.
   */
  def vectorize(document: Array[String], wordList: List[String]): linalg.Vector = {
    val _document = document.map(_.toLowerCase)
    val _wordList = wordList.map(_.toLowerCase).sorted

    val initialWordCounts = scala.collection.mutable.Map[String, Double]()
    _wordList.foreach(word => initialWordCounts(word) = 0.0)

    val wordCounts = _document.foldLeft(initialWordCounts) { (acc, word) =>
      if (acc.contains(word)) {
        acc(word) = acc(word) + 1.0
      } else if (_wordList contains word) {
        acc(word) = 1.0
      }

      acc
    }

    val wordCountsSorted = ListMap(wordCounts.toSeq.sortBy(_._1): _*)

    Vectors.dense(wordCountsSorted.values.toArray)
  }
}

/**
 * Helper to build and evaluate logistic regression models based on their confusion matrix.
 *
 * @param training The training set to be used to build the logistic regression model.
 * @param test The test set used for the evaluation the model.
 */
case class LogRegHelper(training: RDD[LabeledPoint], test: RDD[LabeledPoint]) {
  private def evaluateModel(predictionAndLabels: RDD[(Double, Double)], msg: String) = {
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val cfMatrix = metrics.confusionMatrix

    println(msg)

    printf(
      s"""
           |=================== Confusion matrix ==========================
           |          | %-15s                     %-15s
           |----------+----------------------------------------------------
           |Actual = 0| %-15f                     %-15f
           |Actual = 1| %-15f                     %-15f
           |===============================================================
         """.stripMargin, "Predicted = 0", "Predicted = 1",
      cfMatrix.apply(0, 0), cfMatrix.apply(0, 1), cfMatrix.apply(1, 0), cfMatrix.apply(1, 1))

    cfMatrix.toArray

    val fpr = metrics.falsePositiveRate(0)
    val tpr = metrics.truePositiveRate(0)

    println(
      s"""
       |False positive rate = $fpr
       |True positive rate = $tpr
     """.stripMargin)
  }

  /** Builds and evaluates a logistic regression model based on the BFGS method. */
  def performLogReg() = {
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)

    val predictionAndLabels: RDD[(Double, Double)] = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    evaluateModel(predictionAndLabels, "Logistic regression using BFGS method.")
  }

  /** Builds and evaluates a logistic regression model based on weights found with using R's glm-method on the data. */
  def performLogRegWithWeightsGiven() = {
    val weights: Vector = Vectors.dense(0.833746175771938, -8.14785021875383, 0.262432330501285, -1.99937910634496,
      0.0701712890615536, -0.0655936191264802, -0.408980369055153, -0.64735376719599, 0.187689073816036,
      0.444532778573028, -1.82185138663877, -0.299199000541482, 2.31516305041443, -1.20272016568996,
      -0.986122187022145, -2.64252259577002, 0.463299917001461, -0.173437185656004, -3.91068762983306,
      1.35965422940282, 1.52015523414972, 1.83265596455018, 1.81824680990771, -0.313131391452917, 0.934396040099803,
      -5.40867555026167, -2.17498569362759, -1.34870602549693, 1.00493800384595, -17.7159890027605,
      -0.930304977242082, -6.87226529722958, 0.0855885591427657, -0.804628085850171, -17.7460708574958,
      0.15799019189578, -0.782987768162063, -1.08898113635445, 0.413651064428496, -0.612556409851058,
      -0.461391500104352, -1.51467410854431, -2.4728228258393, -0.555290998510442, -1.55495639417079,
      -14.0790329276713, 0.401977714192125, -0.749594476917577, 3.35723787518922, -3.38910148089078,
      -1.84194781827946, 1.17692030442642, 1.28164263121303, -0.859727559886968, 0.592687850763686,
      -21.1746482611403, 1.222326460073, 0.105447941207522, 1.98943383260695, 0.154573763624215,
      -8.04888988042938, 0.322575450324963, -2.4713916922247, -0.630249323238556, 1.16676457898518,
      1.38005096840468, 0.992897490203946, 6.92019824684486, 0.172027533353067, 3.11379634062188, -0.52795842985082,
      -0.186240279801438, -16.4342857312628, -0.373329497410705, 0.412183688689398, -6.5135497459829,
      3.6294912352502, -2.50428782239549, -5.12844160686143, 3.60797693773892, -0.292857171731344,
      0.0946066479208479, -11.0074891104762, -0.0831555934189275, 1.74249581253245, 0.394388044202608,
      0.539190182405179, 0.482986971371907, 6.39124882217174, -0.105263235643331, -2.38401364176317,
      -0.254393891600753, -0.180853172979074, -3.47164143230984, -0.0425966559519551, 0.392074116932297,
      -0.976741866808988, 0.0260112523189486, 1.08030356208503, 0.505333487193278, 0.67775870127618, -24.9360052033549)

    val modelFromWeightVect = new LogisticRegressionModel(weights, intercept = -5.70797110359394)

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = modelFromWeightVect.predict(features)
      (prediction, label)
    }

    evaluateModel(predictionAndLabels, "Logistic regression using weights found with R.")
  }
}
/**
 * Executable Spark driver programme. Use VM option -Dspark.master=local in the run
 * configuration as well as $MODULE_DIR$ as working directory.
 */
object SpamClassificationViaLogisticRegression extends App {
  case class LabeledSMSText(label: String, SMSText: String)
  case class LabeledTokenizedSMSText(label: String, tokenizedSMSText: Array[String])
  case class LabeledTFVector(encodedLabel: Int, TFVector: linalg.Vector)

  //  def writeToFile(labeledTfVectors: RDD[LabeledTFVector]) = {
  //    val outputPath = "./src/main/resources/data/tf-vectors.tsv"
  //    val separator = '\t'
  //    val writer = new CSVWriter(new FileWriter(outputPath), separator)
  //
  //    labeledTfVectors.foreach {
  //      labeledTfVector =>
  //        val label = Array(labeledTfVector.encodedLabel.toString)
  //        val features = labeledTfVector.TFVector.toArray.map(_.toString)
  //        val entry = Array.concat(label, features)
  //        writer.writeNext(entry)
  //    }
  //
  //    writer.close()
  //  }

  def getTop100WordsList(labeledTokenizedSmsTexts: RDD[LabeledTokenizedSMSText]): List[String] = {
    val words = labeledTokenizedSmsTexts.flatMap(lbldText => lbldText.tokenizedSMSText)
    val wordCount = words.map(word => (word, 1)).reduceByKey((k, l) => k + l).collect().size
    val wordFrequencyMap: Map[String, Long] = words.countByValue()

    val threshold = 136
    val top100Words = wordFrequencyMap.filter { case (word, freq) => freq >= threshold}
    val top100WordsList: List[String] = top100Words.keySet.toList

    top100WordsList
  }

  val sc = new SparkContext(new SparkConf().setAppName("SpamClassificationViaLogisticRegression"))
  val path = "./src/main/resources/data/sms-spam-collection.tsv"
  val input: RDD[String] = sc.textFile(path)
  val separator = '\t'

  /**
   * Contains the [[smsClassificationWithLogRegr.SpamClassificationViaLogisticRegression.LabeledSMSText]]s
   * that are correctly parsed from the input file.
   */
  val labeledSmsTexts: RDD[LabeledSMSText] = input.map { line =>
    val reader = new CSVReader(new StringReader(line), separator)

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
  val wordList = getTop100WordsList(labeledTokenizedSmsTexts)

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


  //  val outputPath = "./src/main/resources/data/tf-vectors.tsv"
  //  val writer = new CSVWriter(new FileWriter(outputPath), separator)
  //
  //  labeledTFVectors.foreach {
  //    labeledTfVector =>
  //      val label = Array(labeledTfVector.encodedLabel.toString)
  //      val features = labeledTfVector.TFVector.toArray.map(_.toString)
  //      val entry = Array.concat(label, features)
  //      writer.writeNext(entry)
  //  }
  //
  //  writer.close()

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

  val logRegHelper = LogRegHelper(training, test)

  logRegHelper.performLogReg()
  logRegHelper.performLogRegWithWeightsGiven()
}

// TODO:
// 1) Later experiment with TF-IDF vectors. Would IDF bring improvement?
// 2) Refactor preprocessing.LineCleaner.normalizeEmailAddress