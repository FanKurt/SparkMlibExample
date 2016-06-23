package com.imac;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public final class JavaKMeans {

	static final Pattern SPACE = Pattern.compile(" ");
	static int num = 0;
	static ArrayList<String> result0 = new ArrayList<String>();
	static ArrayList<String> result1 = new ArrayList<String>();
	static ArrayList<String> result2 = new ArrayList<String>();

	public static void main(String[] args) {
		if (args.length < 3) {
			System.err
					.println("Usage: JavaKMeans <input_file> <k> <max_iterations> [<runs>]");
			System.exit(1);
		}
		String inputFile = args[0];
		int k = Integer.parseInt(args[1]);
		int iterations = Integer.parseInt(args[2]);
		int runs = 1;

		if (args.length >= 4) {
			runs = Integer.parseInt(args[3]);
		}
		SparkConf sparkConf = new SparkConf().setAppName("JavaKMeans");
		
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		JavaRDD<String> lines = sc.textFile(inputFile);
		
		/*
		 * nova-api.log
		 */
		final Pattern nova_log = Pattern
				.compile("([(\\d*)-]*) ([\\d*:]*\\.\\d*) (\\d*) ([A-Z]*) (\\S*) (\\[)(\\S*) (\\S*) (\\S*) ([\\S ]*-]) ([\\d*\\.]*\\d*) (\"\\S*) (\\S*)");
//		
//		for (int i = 0; i < m.groupCount(); i++) {
//			if (i == 4 || i == 5 || i == 7 || i == 8
//					|| i == 9 || i == 11 || i == 12) {
//				mList.add(m.group(i));
//			}
//		}
		
		final Pattern horizon_error_log = Pattern
				.compile("(\\w* \\w* \\d*) (\\S*) (\\d*)] \\S* \\[\\w* (\\d*):\\w* (\\d*)] ((.*))");
		
		final Pattern horizon_log = Pattern
				.compile("([(\\d*).]*) (- -) (\\[[\\S*\\/]*) (\\S*\\]) (\\S*) (.*) (\\S*\\/\\S*) (\\d*) (\\d*) (\"\\S*) (\"\\S*) (\\(.*) (\\S*) (\\(\\S*, \\S* \\S*) (\\S*) ((\\S*))");
		JavaRDD<List<String>> list = lines
				.map(new Function<String, List<String>>() {
					public List<String> call(String line) throws Exception {
						ArrayList<String> mList = new ArrayList<String>();
						Matcher m = horizon_error_log.matcher(line);
						if (m.find()) {
							for (int i = 0; i < m.groupCount(); i++) {
//								if (i == 4 || i == 5 || i == 7 || i == 8
//										|| i == 9 || i == 11 || i == 12) {
									mList.add(m.group(i));
//								}
							}
						}

						return mList;
					}
				});
		JavaRDD<List<String>> vectorArray = list.cache();
		final HashingTF tf = new HashingTF();
		JavaRDD<Vector> points = tf.transform(vectorArray);
		IDF idf = new IDF();
		JavaRDD<Vector> tfIdfs = idf.fit(points).transform(points);
		// final List<Vector> localTfIdfs = tfIdfs.collect();

		// JavaRDD<Vector> points = lines.map(new Function<String, Vector>() {
		// public Vector call(String arg0) throws Exception {
		// String [] token = arg0.split(" ");
		// double [] points = new double[token.length];
		// for(int i=0 ;i< token.length ;i++){
		// points[i] = Double.parseDouble(token[i]);
		// }
		// return Vectors.dense(points);
		// }
		// });
		final KMeansModel model = KMeans.train(points.rdd(), k, iterations,
				runs, KMeans.K_MEANS_PARALLEL());

		List<String> resultList = points.map(new Function<Vector, String>() {
			public String call(Vector v) throws Exception {
				return model.predict(v) + "";
			}
		}).collect();

		Broadcast<List<String>> resultBroadcast = sc.broadcast(resultList);
		final List<String> broadcastList = resultBroadcast.value();
		
		List<String> linesList = lines.collect();
		
		for(String arg0 : linesList){
			if (broadcastList.get(num).equals("0")) {
				result0.add(arg0);
			} else if (broadcastList.get(num).equals("1")) {
				result1.add(arg0);
			} else if (broadcastList.get(num).equals("2")) {
				result2.add(arg0);
			}
			num++;
		}


//		System.out.println( sc.parallelize(result0,1)+"");
//		System.out.println( sc.parallelize(result1,1)+"");
//		System.out.println( sc.parallelize(result2,1)+"");
		 sc.parallelize(result0,1).saveAsTextFile("/cluster/result0");
		 sc.parallelize(result1,1).saveAsTextFile("/cluster/result1");
		 sc.parallelize(result2,1).saveAsTextFile("/cluster/result2");

		// System.out.println("Cluster centers:");
		// for (Vector center : model.clusterCenters()) {
		// System.out.println(" " + center);
		// }
		double cost = model.computeCost(points.rdd());
		System.out.println("Cost: " + cost);
		sc.stop();
	}
}