package preprocessing

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.immutable.ListMap

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