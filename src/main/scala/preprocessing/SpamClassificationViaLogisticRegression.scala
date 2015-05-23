package preprocessing

import java.io.{FileWriter, StringReader}
import java.util.regex._

import scala.collection.Map
import scala.collection.immutable.ListMap
import com.opencsv.{CSVWriter, CSVReader}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg


object LineCleaner {
  type SMSText = String


  def normalizeTemplate(text: SMSText, regex: String, normalizationString: String) = {
    ???
  }

  /**
   * Returns `line` with the following occurrences removed:
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
   * Returns `line` with every occurrence of one of the currency symbol
   * '$', '€' or '£' replaces by the string literal " normalizedcurrencysymbol ".
   */
  def normalizeCurrencySymbol(text: SMSText): SMSText = {
    val regex = "[\\$\\€\\£]"
    val pattern = Pattern.compile(regex)
    val matcher = pattern.matcher(text)

    val cleanedText = matcher.replaceAll(" normalizedcurrencysymbol ")

    cleanedText
  }

  def normalizeEmonicon(text: SMSText): SMSText = {
    val emonicons = List(":-)", ":)", ":D", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)")
    val regex = "(" + emonicons.map(Pattern.quote).mkString("|") + ")"
    val pattern = Pattern.compile(regex)
    val matcher = pattern.matcher(text)

    val cleanedText = matcher.replaceAll(" normalizedemonicon ")

    cleanedText
  }

  /**
   * Returns `line` with every occurrence of one of a number
   * replaced by string literal "normalizednumber".
   */
  def normalizeNumbers(text: SMSText): SMSText = {
    val regex = "\\d+"
    val pattern = Pattern.compile(regex)
    val matcher = pattern.matcher(text)

    val cleanedText = matcher.replaceAll(" normalizednumber ")

    cleanedText
  }


  /**
   * Returns `line` with every occurrence of one of an URL
   * replaced by string literal " normalizedurl ".
   */
  def normalizeURL(text: SMSText): SMSText = {
    val cleanedText = text.split(" ").map { token =>
      if (token.startsWith("www.") || token.startsWith("http://") || token.startsWith("https://")) {
        " normalizedurl "
      } else {
        token
      }
    }.mkString(" ")

    cleanedText
  }

  /**
   * Returns `line` with every occurrence of one of an email address
   * replaced by string literal " normalizedemailadress ".
   */
  def normalizeEmailAddress(text: SMSText): SMSText = {
    val cleanedText = text.split(" ").map { token =>
      if (token.contains("@")) {
        " normalizedemailadress "
      } else {
        token
      }
    }.mkString(" ")

    cleanedText
  }

  /**
   * Returns `line` with HTML character entities removed.
   */
  def removeHTMLCharacterEntities(text: SMSText): SMSText = {
    val HTMLCharacterEntities = List(/*"&nbsp;",*/ "&lt;", "&gt;", "&amp;", "&cent;", "&pound;", "&yen;",
      "&euro;", "&copy;", "&reg;")
    val regex = "(" + HTMLCharacterEntities.map(x => "\\" + x).mkString("|") + ")"
    val pattern = Pattern.compile(regex)
    val matcher = pattern.matcher(text)

    val cleanedText = matcher.replaceAll("")

    cleanedText
  }

  def stem(text: SMSText): SMSText = {
    ???
  }

  def clean(text: SMSText): SMSText = {
    List(text)
      .map(text => text.toLowerCase())
      .map(LineCleaner.normalizeEmonicon)
      .map(LineCleaner.normalizeURL)
      .map(LineCleaner.normalizeEmailAddress)
      .map(LineCleaner.normalizeCurrencySymbol)
      .map(LineCleaner.removeHTMLCharacterEntities)
      .map(LineCleaner.normalizeNumbers)
      .map(LineCleaner.removePunctuationAndSpecialChars)
      .head
  }
}

object DocumentVectorizer {
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

case class LogRegHelper(training: RDD[LabeledPoint], test: RDD[LabeledPoint]) {
  def evaluateModel(predictionAndLabels: RDD[(Double, Double)], msg: String) = {
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

  def performLogReg() = {
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)

    val predictionAndLabels: RDD[(Double, Double)] = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    evaluateModel(predictionAndLabels, "Logistic regression using BFGS method.")
  }

  def performLogRegWithWeightsGiven() = {
    val weights: Vector = Vectors.dense(
      0.846188136825261, -5.18817119544996, -0.0603825085891041, -0.370802595044446,
      0.106146184771286, 0.131813801370727, -0.33219182055294, -0.251976582842262,
      1.47835111947401, 0.0844114904554783, -0.735542132664075, 0.625040867988052,
      2.26379708159678, -0.452403718205055, -0.320094958360598, -2.32127125213211,
      0.419040871943417, 0.0239397680562398, -2.57118870620323, 0.692364403091971,
      1.1923670631835, 1.57869641218471, 1.50873878844324, -0.0283106820897681,
      -0.691660460338901, -0.718692609792941, -2.4149377098733, -0.685288881242416,
      1.15350919054951, -17.4415926553067, -1.30034663531021, -3.46794642436138,
      -0.560918156760458, -0.965430430458044, -17.6362101522642, -0.0443338747000637,
      -2.18325400504607, -1.19434224227548, 0.507924581548693, -0.348731429359891,
      -1.77830959306006, -1.2469989638301, -1.36096379814969, -0.080658893753968,
      -0.173687598811924, -15.4443590375355, 0.171118296762664, -2.7611687451968,
      3.29754789745628, -2.09401166501395, -0.227482570481864, 0.822697583566979,
      1.64748311454022, -0.651905841784664, 0.534642164277911, -16.7633729206904,
      1.19838281729727, -0.531649248073125, 1.33397580612831, -0.00115810680484342,
      -3.9082208822064, -0.0949295634720226, -2.62838212012419, -1.17267535297808,
      0.991355936571525, 0.80421085142069, 0.372400880124102, 5.50186821378904,
      0.604737011663758, 3.076788369929, 0.545664519147207, 0.690518707161155,
      -16.0297923009408, 0.125980861704569, -0.631416941560258, -4.53455579443458,
      2.85945405658738, -2.84304755007709, -1.15070103050509, 3.38591670870141,
      -0.11624183957874, 0.115178943623855, -5.02178453252912, 0.0965672986025181,
      0.732412186988545, 0.63202648542262, 0.46474496862942, 0.744793049081746,
      5.91141929686634, -0.369280546818945, -1.16142559406819, -0.303337374470385,
      0.891466048754432, -7.72594354407987, -0.027605494608393, 0.26032049423333,
      0.431296003329938, 0.617755031912166, 1.36930879186566, 0.146628014258258,
      0.777217750538689, -23.7157831129643
    )
    val modelFromWeightVect = new LogisticRegressionModel(weights, intercept = -5.38059943249913)

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = modelFromWeightVect.predict(features)
      (prediction, label)
    }

    evaluateModel(predictionAndLabels, "Logistic regression using weights found with R.")
  }
}

object SpamClassificationViaLogisticRegression {
  case class LabeledSMSText(label: String, SMSText: String)
  case class LabeledTokenizedSMSText(label: String, tokenizedSMSText: Array[String])
  case class LabeledTFVector(encodedLabel: Int, TFVector: linalg.Vector) {
    def writeToFile(labeledTfVectors: RDD[LabeledTFVector]) = {
      val outputPath = "./src/main/resources/data/tf-vectors.tsv"
      val separator = '\t'
      val writer = new CSVWriter(new FileWriter(outputPath), separator)

      labeledTfVectors.foreach {
        labeledTfVector =>
          val label = Array(labeledTfVector.encodedLabel.toString)
          val features = labeledTfVector.TFVector.toArray.map(_.toString)
          val entry = Array.concat(label, features)
          writer.writeNext(entry)
      }

      writer.close()
    }
  }

  def getTop100WordsList(labeledTokenizedSmsTexts: RDD[LabeledTokenizedSMSText]): List[String] = {
    val words = labeledTokenizedSmsTexts.flatMap(lbldText => lbldText.tokenizedSMSText)
    val wordCount = words.map(word => (word, 1)).reduceByKey((k, l) => k + l).collect().size
    val wordFrequencyMap: Map[String, Long] = words.countByValue()
    val top100Words = wordFrequencyMap.filter { case (word, freq) => freq >= 136}
    val top100WordsList: List[String] = top100Words.keySet.toList

    top100WordsList
  }

  def main(args: Array[String]) = {
    val sc = new SparkContext(new SparkConf().setAppName("SpamClassificationViaLogisticRegression"))
    val path = "./src/main/resources/data/sms-spam-collection.tsv"
    val input: RDD[String] = sc.textFile(path)
    val separator = '\t'

    /**
     * Contains the [[preprocessing.SpamClassificationViaLogisticRegression.LabeledSMSText]]s
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
    val labeledTfVectors: RDD[LabeledTFVector] = labeledTokenizedSmsTexts.map {
      labeledTokenizedSmsText =>
        val encodedLabel = if (labeledTokenizedSmsText.label == "ham") 0 else 1
        val TFVector = DocumentVectorizer.vectorize(labeledTokenizedSmsText.tokenizedSMSText, wordList)
        LabeledTFVector(encodedLabel, TFVector)
    }

    /**
     * Contains the data for building the logistic regression model.
     */
    val data: RDD[LabeledPoint] = labeledTfVectors.map {
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
}

// TODO:
// 0) Make a summary statistic to learn how many words are there actually.
// 1) Use the vectors from `labeledTfVectors` as training set for logistic regression.
//    Should I really use sparse vectors (wouldn't be useful in order to work with them also in R?).
//    It is also not clear whether this is good input format for the logistic regression model builder.
//    PRO: It is already there and I do not have to write my own vectorization. Plus you can easily map
//    to dense vectors by writing the code yourself (there is no build-in toDense-whatever implementation;
//    actually dense vectors and sparse vectors are of trait linalg.Vector and not distinguished by their type).
// 2) Later experiment with TF-IDF vectors.
// 3) Refactor preprocessing.LineCleaner
// 4) Find better solution to error handling while reading (code looks clumsy).
