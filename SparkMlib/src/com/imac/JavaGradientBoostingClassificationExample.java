package com.imac;

// $example on$
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.mllib.util.MLUtils;
// $example off$

public class JavaGradientBoostingClassificationExample {
  public static void main(String[] args) {
    // $example on$
    SparkConf sparkConf = new SparkConf()
      .setAppName("JavaGradientBoostedTreesClassificationExample");
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

    // Load and parse the data file.
    String datapath = "data/mllib/sample_libsvm_data.txt";
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
    
    // Split the data into training and test sets (30% held out for testing)
    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
    JavaRDD<LabeledPoint> trainingData = splits[0];
    JavaRDD<LabeledPoint> testData = splits[1];

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
    boostingStrategy.setNumIterations(3); // Note: Use more iterations in practice.
    boostingStrategy.getTreeStrategy().setNumClasses(2);
    boostingStrategy.getTreeStrategy().setMaxDepth(5);
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
    boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);
    
    trainingData.foreach(new VoidFunction<LabeledPoint>() {
		public void call(LabeledPoint arg0) throws Exception {
			System.out.println(arg0);
		}
	});

    final GradientBoostedTreesModel model =
      GradientBoostedTrees.train(trainingData, boostingStrategy);

    // Evaluate model on test instances and compute test error
    JavaPairRDD<Double, Double> predictionAndLabel =
      testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
        public Tuple2<Double, Double> call(LabeledPoint p) {
          return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
        }
      });
    Double testErr =
      1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
        @Override
        public Boolean call(Tuple2<Double, Double> pl) {
          return !pl._1().equals(pl._2());
        }
      }).count() / testData.count();
    System.out.println("Test Error: " + testErr);
    System.out.println("Learned classification GBT model:\n" + model.toDebugString());

    // Save and load model
//    model.save(jsc.sc(), "target/tmp/myGradientBoostingClassificationModel");
//    GradientBoostedTreesModel sameModel = GradientBoostedTreesModel.load(jsc.sc(),
//      "target/tmp/myGradientBoostingClassificationModel");
    // $example off$
  }

}