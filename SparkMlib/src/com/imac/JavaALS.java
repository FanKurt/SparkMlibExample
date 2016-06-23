package com.imac;

import java.util.Arrays;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;

/**
 * Example using MLlib ALS from Java.
 */
public final class JavaALS {
	static final Pattern COMMA = Pattern.compile(",");

	public static void main(String[] args) {

		SparkConf sparkConf = new SparkConf().setAppName("JavaALS");
		int rank = 10;
		int iterations = 20;
//		String outputDir = args[1];
		int blocks = -1;
		
		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		JavaRDD<String> lines = sc.textFile(args[0]);
		// 1 1 2

		JavaRDD<Rating> ratings = lines.map(new Function<String, Rating>() {
			public Rating call(String line) {
				String[] tok = COMMA.split(line);
				int x = Integer.parseInt(tok[0]);
				int y = Integer.parseInt(tok[1]);
				double rating = Double.parseDouble(tok[2]);
				return new Rating(x, y, rating);
			}
		});
		MatrixFactorizationModel model = ALS.train(ratings.rdd(), rank,iterations, 0.01, blocks);
		JavaRDD<String>userFeatures = model.userFeatures().toJavaRDD()
				.map(new Function<Tuple2<Object, double[]>, String>() {
					public String call(Tuple2<Object, double[]> element)
							throws Exception {
						return element._1() + ","
								+ Arrays.toString(element._2());
					}
				});

		 JavaRDD<String>productFeatures = model.productFeatures().toJavaRDD()
				.map(new Function<Tuple2<Object, double[]>, String>() {
					@Override
					public String call(Tuple2<Object, double[]> element)
							throws Exception {
						return element._1() + ","
								+ Arrays.toString(element._2());
					}
				});
		 
		 JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
			      new Function<Rating, Tuple2<Object, Object>>() {
			        public Tuple2<Object, Object> call(Rating r) {
			          return new Tuple2<Object, Object>(r.user(), r.product());
			        }
			      }
			    );
		
//		RegressionMetrics mm = new RegressionMetrics(userProducts.rdd());
//		mm.rootMeanSquaredError();
//		mm.meanSquaredError();
		
		 
//		System.out.println("Final user/product features written to "+ outputDir);
		
//		userFeatures.saveAsTextFile(outputDir + "/userFeatures");
//		productFeatures.saveAsTextFile(outputDir + "/productFeatures");
		 
		//推薦商品 
		Rating[] recommendProducts =model.recommendProducts(1, 3);
		for(Rating rate : recommendProducts){
			System.out.println("recommendProducts to  User1 :  "+rate	);
		}
		
		//推薦使用者
		Rating[] recommendUsers =model.recommendUsers(2, 3);
		for(Rating rate : recommendUsers){
			System.out.println("recommendUser to Product2 :  "+rate	);
		}
		
		//取得 使用者 對 商品 的評價
		System.out.println("User3 to Prodcut3 Rating  : "+	model.predict(3, 3));
		
		 
		sc.stop();
		
	}
}