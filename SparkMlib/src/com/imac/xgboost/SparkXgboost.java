package com.imac.xgboost;


import java.util.HashMap;
import java.util.Map;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.scala.spark.XGBoost;
import ml.dmlc.xgboost4j.scala.spark.XGBoostModel;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;


public class SparkXgboost {
	public static void main(String[] args) throws XGBoostError {
		JavaSparkContext sc = new JavaSparkContext();
		String inputTestPath = args[0];
		RDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(sc.sc(), inputTestPath);
		
		
		 //specify parameters
		HashMap<String, Object> params = new HashMap<String, Object>();
	    params.put("eta", 1.0);
	    params.put("max_depth", 2);
	    params.put("silent", 1);
	    params.put("objective", "binary:logistic");
	    
	    XGBoostModel booster = XGBoost.train(inputData, (scala.collection.immutable.Map<String, Object>) params, 1, 1, null, null,true);
//	    booster.predi
	}

}
