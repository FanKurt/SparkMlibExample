package com.imac;

// $example on$
import scala.Tuple2;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
// $example off$
import org.apache.spark.SparkConf;

public class JavaNaiveBayesExample {
	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		// $example on$
		String path = args[0];
		JavaRDD<String> rowRdd = jsc.textFile(path);
		JavaRDD<LabeledPoint> inputData =rowRdd.map(new Function<String, LabeledPoint>() {
			public LabeledPoint call(String arg0) throws Exception {
				String [] token = arg0.split(",");
				String [] slice = token[1].split(" ");
				double [] vectors = new double [slice.length-1];
				for(int i=1;i<slice.length;i++){
					vectors[i-1] = Double.parseDouble(slice[i]);
				}
				return new LabeledPoint(Double.parseDouble(token[0]), Vectors.dense(vectors));
			}
		});
		JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[] { 0.6, 0.4 }, 12345);
		JavaRDD<LabeledPoint> training = tmp[0]; // training set
		JavaRDD<LabeledPoint> test = tmp[1]; // test set
		
		final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0 , "multinomial");
		JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
			public Tuple2<Double, Double> call(LabeledPoint p) {
				return new Tuple2<>(model.predict(p.features()), p.label());
			}
		});
		double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
			public Boolean call(Tuple2<Double, Double> pl) {
				return pl._1().equals(pl._2());
			}
		}).count() / (double) test.count();

		System.out.println(accuracy);
		jsc.stop();
	}
}