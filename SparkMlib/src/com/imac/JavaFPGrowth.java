package com.imac;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.AssociationRules.Rule;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset;
import org.apache.spark.mllib.fpm.FPGrowthModel;

import scala.Tuple2;

public class JavaFPGrowth {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("FP-growth Example");
		JavaSparkContext sc = new JavaSparkContext(conf);
		long start = System.currentTimeMillis();
		String input = args[0];
		JavaRDD<String> data = sc.textFile(input);
		JavaRDD<List<String>> transactions = data.map(new Function<String, List<String>>() {
			public List<String> call(String line) {
				List<String> arrList = new ArrayList<String>();
				String[] tokens = line.split(",");
				String row = tokens[18] + "," + tokens[19] + "," + tokens[20];
				String[] lines = row.split(",");
				if (isFormat(lines) && isCorrect(lines)) {
					for (String value : lines) {
						arrList.add(value);
					}
				}
				return arrList;
			}
		});

		FPGrowth fpg = new FPGrowth().setMinSupport(Double.parseDouble(args[1])).setNumPartitions(10);
		FPGrowthModel<String> model = fpg.run(transactions);

		List<FreqItemset<String>> itemSet = model.freqItemsets().toJavaRDD().take(50);
		

		double minConfidence = 0.1;
		List<Rule<String>> ruleList = model.generateAssociationRules(minConfidence).toJavaRDD().collect();
		long end = System.currentTimeMillis();
		ArrayList<Tuple2<Double,String>> arrayList = new ArrayList<>();
		for (AssociationRules.Rule<String> rule : ruleList) {
			arrayList.add(new Tuple2<Double, String>(rule.confidence(), rule.javaAntecedent() + " => " + rule.javaConsequent()));
//			System.out.println(rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
		}
		
		JavaPairRDD<Double, String> rdd = sc.parallelizePairs(arrayList);
		rdd.sortByKey(false).foreach(new VoidFunction<Tuple2<Double,String>>() {
			public void call(Tuple2<Double, String> arg0) throws Exception {
				System.out.println(arg0._2() + ", " + arg0._1());
			}
		});
		
		System.out.println("total : "+(end-start)/1000+" s");
	}

	private static boolean isFormat(String[] tokens) {
		for (String value : tokens) {
			String [] split = value.split("_");
			if (!isNum(split[1])) {
				return false;
			}
		}
		return true;
	}
	
	private static boolean isCorrect(String[] tokens){
		
		for(String value : tokens){
			String [] split = value.split("_");
			String keyWord = split[1];
			int i=0;
			for(String v : tokens){
				String [] values = v.split("_");
				if(keyWord.equals(values[1])){
					i++;
					if(i>=2){
						return false;
					}
				}
			}
		}
		return true;
	}

	private static boolean isNum(String msg) {
		if (java.lang.Character.isDigit(msg.charAt(0))) {
			return true;
		}
		return false;
	}

}
