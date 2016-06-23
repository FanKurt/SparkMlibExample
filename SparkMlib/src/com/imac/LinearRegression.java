package com.imac;

import java.util.List;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.regression.StreamingLinearRegressionWithSGD;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.SparkConf;

public class LinearRegression {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("Linear Regression Example");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Load and parse the data
		String input = args[0];
		String output = args[1];
		JavaRDD<String> data = sc.textFile(input);
		JavaRDD<LabeledPoint> parsedData = data.map(new Function<String, LabeledPoint>() {
			public LabeledPoint call(String line) {
				String[] parts = line.split(",");
				String[] features = parts[1].split(" ");
				double[] v = new double[features.length];
				for (int i = 0; i < features.length - 1; i++)
					v[i] = Double.parseDouble(features[i]);
				return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
			}
		});

		parsedData.cache();
		//
		// Building the model
		int numIterations = 10;
		final LinearRegressionModel model = LinearRegressionWithSGD.train(JavaRDD.toRDD(parsedData), numIterations);
		//
		// Evaluate model on training examples and compute training error
		JavaRDD<Tuple2<Double, Double>> valuesAndPreds = parsedData.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
			public Tuple2<Double, Double> call(LabeledPoint point) {
				double prediction = model.predict(point.features());
				return new Tuple2<Double, Double>(prediction, point.label());
			}
		}).filter(new Function<Tuple2<Double,Double>, Boolean>() {
			public Boolean call(Tuple2<Double, Double> arg0) throws Exception {
				return arg0._1== arg0._2;
			}
		});

		// List<Tuple2<Double, Double>> list = valuesAndPreds.collect();
		// for(Tuple2<Double, Double> tup : list){
		// System.out.println(tup._1+" ~~~ "+tup._2);
		// }

		double MSE = new JavaDoubleRDD(valuesAndPreds.map(new Function<Tuple2<Double, Double>, Object>() {
			public Object call(Tuple2<Double, Double> pair) {
				return Math.pow(pair._1() - pair._2(), 2.0);
			}
		}).rdd()).mean();
		System.out.println("training Mean Squared Error = " + MSE);

		// Save and load model
		// model.save(sc.sc(), output);
		// LinearRegressionModel sameModel = LinearRegressionModel.load(sc.sc(),
		// output);
	}
}