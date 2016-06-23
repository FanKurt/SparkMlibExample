package com.imac;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.PowerIterationClustering;
import org.apache.spark.mllib.clustering.PowerIterationClusteringModel;

import scala.Tuple3;

public class JavaPIC {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("PIC Example");
		JavaSparkContext sc = new JavaSparkContext(conf);
		String input = args[0];
		String output = args[1];
		// Load and parse the data
		JavaRDD<String> data = sc.textFile(input);
		JavaRDD<Tuple3<Long, Long, Double>> similarities = data
				.map(new Function<String, Tuple3<Long, Long, Double>>() {
					public Tuple3<Long, Long, Double> call(String line) {
						String[] parts = line.split(" ");
						return new Tuple3<>(new Long(parts[0]), new Long(
								parts[1]), new Double(parts[2]));
					}
				});

		// Cluster the data into two classes using PowerIterationClustering
		PowerIterationClustering pic = new PowerIterationClustering().setK(2)
				.setMaxIterations(10);
		PowerIterationClusteringModel model = pic.run(similarities);

		for (PowerIterationClustering.Assignment a : model.assignments()
				.toJavaRDD().collect()) {
			System.out.println(a.id() + " -> " + a.cluster());
		}

		// Save and load model
		model.save(sc.sc(), output);
		PowerIterationClusteringModel sameModel = PowerIterationClusteringModel
				.load(sc.sc(), output);
	}

}
