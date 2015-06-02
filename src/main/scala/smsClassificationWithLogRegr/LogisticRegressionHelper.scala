package smsClassificationWithLogRegr

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Helper to build and evaluate logistic regression models based on their confusion matrix.
 *
 * @param training The training set to be used to build the logistic regression model.
 * @param test The test set used for the evaluation the model.
 */
case class LogisticRegressionHelper(training: RDD[LabeledPoint], test: RDD[LabeledPoint]) {
  private def evaluateModel(predictionAndLabels: RDD[(Double, Double)], msg: String) = {
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val cfMatrix = metrics.confusionMatrix

    println(msg)

    printf(
      s"""
           |=================== Confusion matrix ==========================
           |          | %-15s                     %-15s
           |----------+----------------------------------------------------
           |Actual = 0| %-15f                     %-15f
           |Actual = 1| %-15f                     %-15f
           |===============================================================
         """.stripMargin, "Predicted = 0", "Predicted = 1",
      cfMatrix.apply(0, 0), cfMatrix.apply(0, 1), cfMatrix.apply(1, 0), cfMatrix.apply(1, 1))

    cfMatrix.toArray

    val fpr = metrics.falsePositiveRate(0)
    val tpr = metrics.truePositiveRate(0)

    println(
      s"""
       |False positive rate = $fpr
       |True positive rate = $tpr
     """.stripMargin)
  }

  /** Builds and evaluates a logistic regression model based on the BFGS method. */
  def performLogReg() = {
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)

    val predictionAndLabels: RDD[(Double, Double)] = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    evaluateModel(predictionAndLabels, "Logistic regression using BFGS method.")
  }

  /** Builds and evaluates a logistic regression model based on weights found with using R's glm-method on the data. */
  def performLogRegWithWeightsGiven() = {
    val weights: Vector = Vectors.dense(0.833746175771938, -8.14785021875383, 0.262432330501285, -1.99937910634496,
      0.0701712890615536, -0.0655936191264802, -0.408980369055153, -0.64735376719599, 0.187689073816036,
      0.444532778573028, -1.82185138663877, -0.299199000541482, 2.31516305041443, -1.20272016568996,
      -0.986122187022145, -2.64252259577002, 0.463299917001461, -0.173437185656004, -3.91068762983306,
      1.35965422940282, 1.52015523414972, 1.83265596455018, 1.81824680990771, -0.313131391452917, 0.934396040099803,
      -5.40867555026167, -2.17498569362759, -1.34870602549693, 1.00493800384595, -17.7159890027605,
      -0.930304977242082, -6.87226529722958, 0.0855885591427657, -0.804628085850171, -17.7460708574958,
      0.15799019189578, -0.782987768162063, -1.08898113635445, 0.413651064428496, -0.612556409851058,
      -0.461391500104352, -1.51467410854431, -2.4728228258393, -0.555290998510442, -1.55495639417079,
      -14.0790329276713, 0.401977714192125, -0.749594476917577, 3.35723787518922, -3.38910148089078,
      -1.84194781827946, 1.17692030442642, 1.28164263121303, -0.859727559886968, 0.592687850763686,
      -21.1746482611403, 1.222326460073, 0.105447941207522, 1.98943383260695, 0.154573763624215,
      -8.04888988042938, 0.322575450324963, -2.4713916922247, -0.630249323238556, 1.16676457898518,
      1.38005096840468, 0.992897490203946, 6.92019824684486, 0.172027533353067, 3.11379634062188, -0.52795842985082,
      -0.186240279801438, -16.4342857312628, -0.373329497410705, 0.412183688689398, -6.5135497459829,
      3.6294912352502, -2.50428782239549, -5.12844160686143, 3.60797693773892, -0.292857171731344,
      0.0946066479208479, -11.0074891104762, -0.0831555934189275, 1.74249581253245, 0.394388044202608,
      0.539190182405179, 0.482986971371907, 6.39124882217174, -0.105263235643331, -2.38401364176317,
      -0.254393891600753, -0.180853172979074, -3.47164143230984, -0.0425966559519551, 0.392074116932297,
      -0.976741866808988, 0.0260112523189486, 1.08030356208503, 0.505333487193278, 0.67775870127618, -24.9360052033549)

    val modelFromWeightVect = new LogisticRegressionModel(weights, intercept = -5.70797110359394)

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = modelFromWeightVect.predict(features)
      (prediction, label)
    }

    evaluateModel(predictionAndLabels, "Logistic regression using weights found with R.")
  }
}