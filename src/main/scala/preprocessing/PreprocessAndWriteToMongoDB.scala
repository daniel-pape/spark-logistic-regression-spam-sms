package preprocessing

import java.io.FileReader

import com.mongodb.DBObject
import com.mongodb.casbah.MongoConnection
import com.mongodb.casbah.commons.MongoDBObject
import com.opencsv.CSVReader
import common.Common._
import preprocessing.LineCleaner._

import scala.io.Source._


object MongoFactory {
  import scala.collection.JavaConversions._

  private final val SERVER = "localhost"
  private final val PORT = 27017
  private final val DB_NAME = "spam_sms_db"
  private final val COLLECTION_NAME = "spam_sms"

  private final val connection = MongoConnection(SERVER, PORT)
  private final val collection = connection(DB_NAME)(COLLECTION_NAME)

  def buildDBObject(input: DBInput): DBObject = {
    val builder = MongoDBObject.newBuilder
    builder += "label" -> input.label
    builder += "original_text" -> input.SMSText
    builder += "cleaned_text" -> input.cleanedSMSText
    builder += "tf_vector" -> seqAsJavaList(input.TFVector)
    builder.result()
  }

  def insertDBObject(dbObject: DBObject) = {
    collection += dbObject
  }
}

/**
 * Reads the input TSV-file, pre-processes it and writes the result
 * as a collection of MongoDB documents. Use case: Exchange with R.
 */
object PreprocessAndWriteToMongoDB extends App {
  import preprocessing.MongoFactory._

  val reader: CSVReader  = new CSVReader(new FileReader(path), CSVSeparator)
  val src = fromFile(path)
  val wordList = CreateWordList.wordList

  for (line <- src.getLines()) {
    val Array(label, text) = line.split(CSVSeparator).map(_.trim)

//    printf(s"""%-4s %-100s\n""".stripMargin, label, clean(text))

    val tokenizedSMSText = text.split(" ")
    val TFVectorAsArray = DocumentVectorizer.vectorize(tokenizedSMSText, wordList).toArray
    val dbInput = DBInput(label, text, clean(text), TFVectorAsArray)
    val dbObject = buildDBObject(dbInput)
    insertDBObject(dbObject)
  }
}
