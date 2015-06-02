package Test

import org.apache.spark.mllib.linalg.Vectors
import org.junit.Assert._
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import preprocessing.DocumentVectorizer

@RunWith(classOf[JUnit4])
class DocumentVectorizerTest {
  @Test
  def VectorizationTest1(): Unit = {
    val document = Array("Hello", "this", "is", "a", "test", "One", "two", "three", "test", "test")
    val wordList = List("A", "be", "hello", "is", "one", "test", "this", "three", "two", "zoo")

    val expected = Vectors.dense(Array(1.0, 0.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 0.0))
    val actual = DocumentVectorizer.vectorize(document, wordList)

    assertEquals("`vectorize` should return correct term frequency vector", expected, actual)
  }

  @Test
  def VectorizationTest2(): Unit = {
    val input = "To be or not to be that is the question" + "That is utter rubbish"
    val document = input.split(" ")
    val wordList = List("be", "not", "To", "unused")

    val expected = Vectors.dense(Array(2.0, 1.0, 2.0, 0.0))
    val actual = DocumentVectorizer.vectorize(document, wordList)

    assertEquals("`vectorize` should return correct term frequency vector", expected, actual)
  }

  // TODO: Test with empty document and empty word list!
}