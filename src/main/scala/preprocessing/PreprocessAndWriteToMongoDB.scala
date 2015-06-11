package preprocessing

import java.io.FileReader

import com.mongodb.DBObject
import com.mongodb.casbah.Imports.wrapDBObj
import com.mongodb.casbah.MongoConnection
import com.mongodb.casbah.commons.MongoDBObject
import com.opencsv.CSVReader
import common.Common._
import preprocessing.LineCleaner._

import scala.io.Source._

object MongoFactory {
  private final val SERVER = "localhost"
  private final val PORT = 27017
  private final val DB_NAME = "spam_sms_db"
  private final val COLLECTION_NAME = "spam_sms"

  private final val connection = MongoConnection(SERVER, PORT)
  private final val collection = connection(DB_NAME)(COLLECTION_NAME)

  def toMongoDBObject(ld: labeledData): MongoDBObject = {
    MongoDBObject(
      "label" -> ld.label,
      "original_text" -> ld.SMSText,
      "cleaned_text" -> ld.cleanedSMSText,
      "tf_vector" -> ld.TFVector.mkString(", ")
      // Ignore:
      // import scala.collection.JavaConversions._
      //"tf_vector" -> seqAsJavaList(ld.TFVector)
    )
  }

  def fromMongoDBObject(o: MongoDBObject): labeledData = {
    val label = o.as[String]("label")
    val SMSText = o.as[String]("original_text")
    val cleanedSMSText = o.as[String]("cleaned_text")
    val tfVector = o.as[String]("tf_vector").split(", ").map(_.toDouble)
    // Ignore:
    // val tfVector: MongoDBList = o.as[MongoDBList]("tf_vector")

    labeledData(label, SMSText, cleanedSMSText, tfVector)
  }

  def insertMongoDBObject(o: MongoDBObject) = {
    collection += o.underlying
  }

  def printCollection() = {
    val cursor = collection.find()
    val dbObjects: Vector[DBObject] = cursor.toVector
    val mongoDBObjects = dbObjects
    mongoDBObjects.foreach(o => println(fromMongoDBObject(o)))
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

    // printf(s"""%-4s %-100s\n""".stripMargin, label, clean(text))

    val tokenizedSMSText = text.split(" ")
    val TFVectorAsArray = DocumentVectorizer.vectorize(tokenizedSMSText, wordList).toArray
    val dbInput = labeledData(label, text, clean(text), TFVectorAsArray)
    val dbObject = toMongoDBObject(dbInput)
    insertMongoDBObject(dbObject)
  }

  printCollection()
}
