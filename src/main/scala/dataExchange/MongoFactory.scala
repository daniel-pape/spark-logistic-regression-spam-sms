package dataExchange

import com.mongodb.DBObject
import com.mongodb.casbah.Imports.wrapDBObj
import com.mongodb.casbah.MongoConnection
import com.mongodb.casbah.commons.MongoDBObject
import common.Common.{MongoDBConf, labeledData}

/**
 * Provides access to the MongoDB collection.
 */
object MongoFactory {
  private final val connection = MongoConnection(MongoDBConf.server, MongoDBConf.port)
  private final val collection = connection(MongoDBConf.dbName)(MongoDBConf.collectionName)

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