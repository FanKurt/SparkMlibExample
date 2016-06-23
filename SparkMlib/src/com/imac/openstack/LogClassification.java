package com.imac.openstack;

import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import org.json.simple.JSONObject;

import scala.Tuple2;

public class LogClassification {
	private static org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger(LogClassification.class);
	public static void main(String[] args) {
		SparkConf conf = new SparkConf();
		conf.set("es.index.auto.create", "true");
		conf.set("es.nodes", "10.26.1.9:9200");
		conf.set("es.resource", "ceph-log/logs");
//		conf.set("es.query", args[0]);
		conf.set("es.input.json", "true");
		conf.setAppName("CephLogAnalysis");

		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc);
		
		
		JavaRDD<LabeledPoint> jsonRDD = esRDD.map(new Function<Tuple2<String,Map<String,Object>>, LabeledPoint>() {
			public LabeledPoint call(Tuple2<String, Map<String, Object>> arg0) throws Exception {
				JSONObject jsonObject = new JSONObject(arg0._2);
				double[] array = new double[8];
				try{
					array[0]= Double.parseDouble(jsonObject.get("pgmap").toString());
					array[1]= Double.parseDouble(jsonObject.get("pgs").toString());
					array[2]= Double.parseDouble(jsonObject.get("data").toString());
					array[3]= Double.parseDouble(jsonObject.get("used").toString());
					array[4]= Double.parseDouble(jsonObject.get("unused").toString());
					array[5]= Double.parseDouble(jsonObject.get("total").toString());
					array[6]= Double.parseDouble(jsonObject.get("rd").toString());
					array[7]= Double.parseDouble(jsonObject.get("wr").toString());
					double ops =  Double.parseDouble(jsonObject.get("ops").toString());
					return new LabeledPoint((ops>133)?0:1, Vectors.dense(array));
				}catch(Exception e){
					return  new LabeledPoint(100,null);
				}
			}
		}).filter(new Function<LabeledPoint, Boolean>() {
			public Boolean call(LabeledPoint arg0) throws Exception {
				return arg0.label()<=1;
			}
		});
		
		
		JavaRDD<LabeledPoint>[] splitRDD = jsonRDD.randomSplit(new double [] {0.7,0.3});
		
		JavaRDD<LabeledPoint> trainRDD = splitRDD[0];
		JavaRDD<LabeledPoint> testRDD = splitRDD[1];
		
//		final NaiveBayesModel model = NaiveBayes.train(trainRDD.rdd(), 1.0);
		
//		int numIterations = 100;
//	    final SVMModel model = SVMWithSGD.train(trainRDD.rdd(), numIterations);
		
		BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
	    boostingStrategy.setNumIterations(100); // Note: Use more iterations in practice.
	    boostingStrategy.getTreeStrategy().setNumClasses(2);
	    boostingStrategy.getTreeStrategy().setMaxDepth(10);
	    final GradientBoostedTreesModel model = GradientBoostedTrees.train(trainRDD, boostingStrategy);
		
//		Integer numClasses = 2;
//		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
//		String impurity = "gini";
//		Integer maxDepth = 5;
//		Integer maxBins = 32;
//		final DecisionTreeModel model = DecisionTree.trainClassifier(trainRDD, numClasses,
//		  categoricalFeaturesInfo, impurity, maxDepth, maxBins);
		
//		final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
//	      .setNumClasses(2)
//	      .run(trainRDD.rdd());
		
		JavaRDD<Tuple2<Object, Object>> test_rdd = testRDD.map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
			public Tuple2<Object, Object> call(LabeledPoint arg0) throws Exception {
				return new Tuple2<Object, Object>(model.predict(arg0.features()), arg0.label());
			}
		});
		
		
		Double accuracy =test_rdd.filter(new Function<Tuple2<Object,Object>, Boolean>() {
			public Boolean call(Tuple2<Object, Object> arg0) throws Exception {
				return arg0._1().equals(arg0._2());
			}
		}).count()/(double)test_rdd.count();
		
	     
	    System.out.println("accuracy  :  "+accuracy);
		
	    BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(test_rdd));
	    double auROC = metrics.areaUnderROC();
	    	    

	    System.out.println("Area under ROC = " + auROC);
	    sc.stop();
		
	}
}
