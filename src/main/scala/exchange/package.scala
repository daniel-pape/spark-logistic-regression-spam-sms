import com.mongodb.hadoop.{MongoInputFormat, MongoOutputFormat}
import org.bson.BSONObject

package object exchange {
  final val keyClass = classOf[Object]
  final val valueClass = classOf[BSONObject]
  final val inputFormatClass = classOf[MongoInputFormat]
  final val outputFormatClass = classOf[MongoOutputFormat[_, _]]
}
