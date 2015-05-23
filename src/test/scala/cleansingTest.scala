package Test

import com.google.common.collect.ImmutableMap
import org.apache.spark.mllib.linalg.Vectors
import org.junit.Assert._
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import preprocessing.DocumentVectorizer

import preprocessing.LineCleaner._

import scala.collection.immutable.ListMap

@RunWith(classOf[JUnit4])
class Basics_Test {
  @Test
  def test1: Unit = {
    val actual = normalizeCurrencySymbol("$")
    val desired = List("label", "normalizedcurrencysymbol")
    assertEquals("`normalizeCurrencySymbol` should replace $.", actual, desired)
  }

  @Test
  def test2: Unit = {
    val actual = normalizeCurrencySymbol("€")
    val desired = List("label", "normalizedcurrencysymbol")
    assertEquals("`normalizeCurrencySymbol` should replace €.", actual, desired)
  }

  @Test
  def test3: Unit = {
    val actual = normalizeCurrencySymbol("£")
    val desired = List("label", "normalizedcurrencysymbol")
    assertEquals("`normalizeCurrencySymbol` should replace £.", actual, desired)
  }

  @Test
  def test4: Unit = {
    val text = HTMLCharacterEntities.mkString("")

    val expected = removeHTMLCharacterEntities(text)
    val actual = List("label", "")

    assertEquals("`removeHTMLCharacterEntities` should remove all HTML character entities.", expected, actual)
  }

  @Test
  def test5: Unit = {
    val document = Array("Hello", "this", "is", "a", "test", "One", "two", "three", "test", "test")
    val wordList = List("A", "be", "hello", "is", "one", "test", "this", "three", "two", "zoo")

    val expected = Vectors.dense(Array(1.0, 0.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 0.0))
    val actual = DocumentVectorizer.vectorize(document, wordList)

    assertEquals("`vectorize` should return correct term frequency vector", expected, actual)
  }

  @Test
  def test6: Unit = {
    val input = "To be or not to be that is the question" + "That is utter rubbish"
    val document = input.split(" ")
    val wordList = List("be", "not", "To", "unused")

    val expected = Vectors.dense(Array(2.0, 1.0, 2.0, 0.0))
    val actual = DocumentVectorizer.vectorize(document, wordList)

    assertEquals("`vectorize` should return correct term frequency vector", expected, actual)
  }

  // TODO: Test with empty document and empty word list!
}