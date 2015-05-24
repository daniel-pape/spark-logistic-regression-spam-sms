import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object logisticRegressionExample extends App {
  val sc = new SparkContext(new SparkConf().setAppName("LogisticRegression"))

  val path = "/home/daniel/IdeaProjects/logistic-regression-spark/src/main/resources/data/sample_libsvm_data.txt"
  val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, path)

  // Split data into training (60%) and test (40%).
  val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
  val training = splits(0).cache()
  val test = splits(1)

  // Run training algorithm to build the model
  val model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training)


  // Compute raw scores on the test set.
  val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
    val prediction = model.predict(features)
    (prediction, label)
  }

  // Get evaluation metrics.
  val metrics = new MulticlassMetrics(predictionAndLabels)
  val precision = metrics.precision
  println("Precision = " + precision)

  // Save and load model
  model.save(sc, "myModelPath")
  val sameModel = LogisticRegressionModel.load(sc, "myModelPath")
}

object logisticRegressionBFGS extends App {
  val sc = new SparkContext(new SparkConf().setAppName("LogisticRegression"))

  val path = "/home/daniel/IdeaProjects/logistic-regression-spark/src/main/resources/data/admission-data.csv"
  val input: RDD[String] = sc.textFile(path)

  //input.foreach(println)

  val data: RDD[LabeledPoint] = input.map { line =>
    val parts = line.split(',')
    val label = java.lang.Double.parseDouble(parts(0))
    val features = Array(parts(1), parts(2)).map(_.trim()).map(java.lang.Double.parseDouble)
    LabeledPoint(label, Vectors.dense(features))
  }

  //data.foreach(println)

  // Split data into training (75%) and test (25%).
  val splits = data.randomSplit(Array(0.75, 0.25), seed = 11L)
  val training = splits(0).cache()
  val test = splits(1)

  val model = new LogisticRegressionWithLBFGS().run(training)

  val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
    val prediction = model.predict(features)
    (prediction, label)
  }

  //predictionAndLabels.foreach(println)

  val metrics = new MulticlassMetrics(predictionAndLabels)
  val precision = metrics.precision
  val recall = metrics.recall
  val fpr = metrics.falsePositiveRate(0)
  val tpr = metrics.truePositiveRate(0)

  println(
    s"""
       |Precision = $precision
       |Recall = $recall
       |fpr = $fpr
       |tpr = $tpr
     """.stripMargin)

  // TODO: Can I overwrite the string method of the cfMatrix?

  val cfMatrix = metrics.confusionMatrix
  println(cfMatrix)
}

object logisticRegression extends App {
  val sc = new SparkContext(new SparkConf().setAppName("LogisticRegression"))

  val path = "/home/daniel/IdeaProjects/logistic-regression-spark/src/main/resources/data/admission-data.csv"
  val input: RDD[String] = sc.textFile(path)

  //input.foreach(println)

  val data: RDD[LabeledPoint] = input.map { line =>
    val parts = line.split(',')
    val label = java.lang.Double.parseDouble(parts(0))
    val features = Array(parts(1), parts(2)).map(_.trim()).map(java.lang.Double.parseDouble)
    LabeledPoint(label, Vectors.dense(features))
  }

  //data.foreach(println)

  // Split data into training (75%) and test (25%).
  val splits = data.randomSplit(Array(0.75, 0.25), seed = 11L)
  val training = splits(0).cache()
  val test = splits(1)

  val weights: Vector = Vectors.dense(0.2194497, 0.2246669)
  val model = new LogisticRegressionModel(weights, intercept = -27.1881248)

  val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
    val prediction = model.predict(features)
    (prediction, label)
  }

  println(s"Used threshold: model.getThreshold")

  //predictionAndLabels.foreach(println)

  val metrics = new MulticlassMetrics(predictionAndLabels)
  val precision = metrics.precision
  val recall = metrics.recall
  val fpr = metrics.falsePositiveRate(0)
  val tpr = metrics.truePositiveRate(0)


  println(
    s"""
       |Precision = $precision
       |Recall = $recall
       |fpr = $fpr
       |tpr = $tpr
     """.stripMargin)

  // TODO: Can I overwrite the string method of the cfMatrix?

  val cfMatrix = metrics.confusionMatrix
  println(cfMatrix)

}
