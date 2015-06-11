package preprocessing

import java.io.FileReader

import com.opencsv.CSVReader
import common.Common._
import preprocessing.LineCleaner._

import scala.io.Source._

/**
 * Reads the input TSV-file, pre-processes it and writes the result
 * as a collection of MongoDB documents. Use case: Exchange with R.
 */
object PreprocessAndWriteToMongoDB extends App {
  import dataExchange.MongoFactory._

  val reader: CSVReader  = new CSVReader(new FileReader(path), CSVSeparator)
  val src = fromFile(path)
  val wordList = CreateWordList.wordList

  for (line <- src.getLines()) {
    val Array(label, text) = line.split(CSVSeparator).map(_.trim)

    // printf(s"""%-4s %-100s\n""".stripMargin, label, clean(text))

    val tokenizedSMSText = text.split(" ")
    val TFVectorAsArray = DocumentVectorizer.vectorize(tokenizedSMSText, wordList).toArray
    val dbInput = labeledData(label, text, clean(text), TFVectorAsArray)
    val dbObject = toMongoDBObject(dbInput)
    insertMongoDBObject(dbObject)
  }

  // printCollection()
}
