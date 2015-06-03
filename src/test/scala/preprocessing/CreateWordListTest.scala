package preprocessing

import org.junit.Assert._
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4
import preprocessing.CreateWordList.createWordFrequencyMap

@RunWith(classOf[JUnit4])
class CreateWordListTest {
  @Test
  def CreateWordFrequencyMapCountsCorrectly1() = {
    val words = List("a", "a", "a", "b", "b", "c")
    val actual = createWordFrequencyMap(words)
    val desired = Map("a" -> 3, "b" -> 2, "c" -> 1)

    val msg = "Word frequency map should count occurrences of words correctly."
    assertEquals(msg, actual, desired)
  }

  @Test
  def CreateWordFrequencyMapCountsCorrectly2() = {
    val words = List("a", "b", "c")
    val actual = createWordFrequencyMap(words)
    val desired = Map("a" -> 1, "b" -> 1, "c" -> 1)

    val msg = "Word frequency map should count occurrences of words correctly."
    assertEquals(msg, actual, desired)
  }

  @Test
  def WordFrequencyIsIndependentOfOrder() = {
    val words = List("a", "a", "a", "b", "b", "c").reverse
    val actual = createWordFrequencyMap(words)
    val desired = Map("a" -> 3, "b" -> 2, "c" -> 1)

    val msg = "Word frequency map should count occurrences of words correctly independent of their order."
    assertEquals(msg, actual, desired)
  }

  @Test
  def CreateWordFrequencyMapForNoWordsShouldBeEmpty() = {
    val words = List()
    val actual = createWordFrequencyMap(words)
    val desired = Map()

    val msg = "Word frequency map given no words should be empty."
    assertEquals(msg, actual, desired)
  }
}
