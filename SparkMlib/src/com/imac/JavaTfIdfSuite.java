package com.imac;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

public class JavaTfIdfSuite {
	private static JavaSparkContext sc;

	public static void main(String[] args) {
		sc = new JavaSparkContext("local", "JavaTfIdfSuite");
//		tfIdf();
		tfIdfMinimumDocumentFrequency();
		sc.stop();
		sc = null;
	}

	public static void tfIdf() {
		// The tests are to check Java compatibility.
		HashingTF tf = new HashingTF();
		JavaRDD<List<String>> documents = sc.parallelize(Arrays.asList(
				Arrays.asList("this this thisis a sentence".split(" ")),
				Arrays.asList("this this this is another sentence".split(" ")),
				Arrays.asList("is still a sentence".split(" "))), 2);

		JavaRDD<Vector> termFreqs = tf.transform(documents);

		// termFreqs.collect();
		IDF idf = new IDF();
		JavaRDD<Vector> tfIdfs = idf.fit(termFreqs).transform(termFreqs);
		List<Vector> localTfIdfs = tfIdfs.collect();
		int indexOfThis = tf.indexOf("this");
		for (Vector v : localTfIdfs) {
			System.out.println("this   " + v.apply(indexOfThis));
			// Assert.assertEquals(0.0, v.apply(indexOfThis), 1e-15);
		}
	}

	public static void tfIdfMinimumDocumentFrequency() {
		// The tests are to check Java compatibility.
		final HashingTF tf = new HashingTF();
		final JavaRDD<List<String>> documents = sc.parallelize(Arrays.asList(
				Arrays.asList("this is a sentence".split(" ")),
				Arrays.asList("that is another sentence".split(" ")),
				Arrays.asList("these are still a sentence".split(" "))), 2);
		JavaRDD<Vector> termFreqs = tf.transform(documents);
		termFreqs.collect();
		IDF idf = new IDF(2);
		JavaRDD<Vector> tfIdfs = idf.fit(termFreqs).transform(termFreqs);
		
		
		final List<Vector> vectorList = tfIdfs.collect();
		
		
		JavaPairRDD<String, Double> aa = documents.flatMapToPair(new PairFlatMapFunction<List<String>, String, Double>() {
			public Iterable<Tuple2<String, Double>> call(List<String> arg0)
					throws Exception {
				ArrayList<Tuple2<String, Double>> arrList = new ArrayList<Tuple2<String,Double>>();
				for(String v : arg0){
					int indexOfThis = tf.indexOf(v);
					for(Vector vector : vectorList){
						arrList.add(new Tuple2<String, Double>(v , vector.apply(indexOfThis)));
					}
				}
				return arrList;
			}
		}).filter(new Function<Tuple2<String,Double>, Boolean>() {
			public Boolean call(Tuple2<String, Double> arg0) throws Exception {
				return (arg0._2 > 0.0);
			}
		}).distinct();
		
		for(Tuple2<String, Double> v : aa.collect()){
			System.out.println(v);
		}
		

	}

}