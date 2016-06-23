package com.imac.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;

public class Test {

	public static void main(String[] args) {
		JavaSparkContext sc = new JavaSparkContext();
		ArrayList<Double> arrayList = new ArrayList<Double>();
		arrayList.add(1.0);
		arrayList.add(2.0);
		arrayList.add(3.0);
		
		JavaRDD<Double> rdd = sc.parallelize(arrayList);
		
		Accumulator<Double> mse =sc.accumulator(0.0);
		Accumulator<Double> mae =sc.accumulator(0.0);
		
		Tuple2<RDD<Double>, RDD<Double>>[] ml = MLUtils.kFold(rdd.rdd(), 3, 10,rdd.classTag());
		
		for(Tuple2<RDD<Double>, RDD<Double>> v : ml){
			double b = 0 ;
			for(Double vv : v._1.toJavaRDD().collect()){
				System.out.println("1 "+vv);
				b+=vv;
			}
			System.out.println(""+b);
			System.out.println("--------------------");
			
			mse.add(b);
//			for(Double vv : v._2.toJavaRDD().collect()){
//				System.out.println("2 "+vv);
//			}
//			System.out.println("--------------------");
		}
		System.out.println("-------------------- total "+mse.value());
		System.out.println("-------------------- average "+mse.value()/ml.length);
	}
	
}
