package preprocessing

import java.io.FileWriter

import com.opencsv.CSVWriter
import common.Common._
import org.apache.spark.rdd.RDD

object PreprocessAndWriteToFile {
  def writeToFile(labeledTfVectors: RDD[LabeledTFVector]) = {
    val outputPath = "./src/main/resources/data/tf-vectors.tsv"
    val writer = new CSVWriter(new FileWriter(outputPath), CSVSeparator)

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
