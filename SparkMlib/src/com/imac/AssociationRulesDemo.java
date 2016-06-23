package com.imac;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset;

public class AssociationRulesDemo {

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("AssociationRules Example");
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		JavaRDD<FPGrowth.FreqItemset<String>> freqItemsets = sc
				.parallelize(Arrays.asList(new FreqItemset<String>(
						new String[] { "a" }, 15L), new FreqItemset<String>(
						new String[] { "b" }, 35L), new FreqItemset<String>(
						new String[] { "a", "b" }, 12L)));

		AssociationRules arules = new AssociationRules().setMinConfidence(0.8);
		JavaRDD<AssociationRules.Rule<String>> results = arules
				.run(freqItemsets);

		for (AssociationRules.Rule<String> rule : results.collect()) {
			System.out.println(rule.javaAntecedent() + " => "
					+ rule.javaConsequent() + ", " + rule.confidence());
		}
	}

}
