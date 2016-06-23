package com.imac.openstack;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.TimeZone;

import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import org.json.simple.JSONObject;

import scala.Tuple2;

public class OpenStackLogAnalysis {
	private static org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger(OpenStackLogAnalysis.class);
	private static SimpleDateFormat dateFormatUtc = new SimpleDateFormat("EE MMM dd HH:mm:ss zzz yyyy", Locale.US);

	public static void main(String[] args) throws ParseException {
		dateFormatUtc.setTimeZone(TimeZone.getTimeZone("UCT"));
		
		SparkConf conf = new SparkConf();
		conf.set("es.index.auto.create", "true");
		conf.set("es.nodes", "10.26.1.9:9200");
		conf.set("es.resource", "ceph-log/logs");
//		conf.set("es.query", args[0]);
		conf.set("es.input.json", "true");
		conf.setAppName("CephLogAnalysis");

		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc);
		
//		//計算每日平均資料寫入流量
//		JavaPairRDD<String, Object> rawRDD = esRDD.mapToPair(new PairFunction<Tuple2<String,Map<String,Object>>, String, Object>() {
//			public Tuple2<String, Object> call(Tuple2<String, Map<String, Object>> arg0) throws Exception {
//				JSONObject jsonObject = new JSONObject(arg0._2);
//				try{
//					String used = jsonObject.get("used").toString();
//					String date = jsonObject.get("date").toString();
//					if(Double.parseDouble(used)>0){
//						return new Tuple2<String, Object>(date, Double.parseDouble(used));
//					}else{
//						return null;
//					}
//			
//				}catch(Exception e){
//					return null;
//				}
//			}
//		}).filter(new Function<Tuple2<String,Object>, Boolean>() {
//			public Boolean call(Tuple2<String, Object> arg0) throws Exception {
//				return arg0!=null && isWorkingDay(arg0._1());
//			}
//		});
//		
//		
//		Map<Object, String> map = rawRDD.mapToPair(new PairFunction<Tuple2<String,Object>, Object, String>() {
//			public Tuple2<Object, String> call(Tuple2<String, Object> arg0) throws Exception {
//				return arg0.swap();
//			}
//		}).collectAsMap();
//		
//		JavaRDD<Object> used_rdd = rawRDD.values();
//		
//		double max = new JavaDoubleRDD(used_rdd.rdd()).max();
//		double min = new JavaDoubleRDD(used_rdd.rdd()).min();
//		
//		double dayCount =(double) getDayCount(map.get(min), map.get(max));
//		System.out.println("days "+ dayCount);
//		final double wr_average = (max-min)/dayCount;
//		
//		System.out.println("Average : "+wr_average);
		final double wr_average =20.130434782608695;
		
		//資料準備
		JavaRDD<LabeledPoint> jsonRDD = esRDD.filter(new Function<Tuple2<String,Map<String,Object>>, Boolean>() {
			public Boolean call(Tuple2<String, Map<String, Object>> arg0) throws Exception {
				try{
					return arg0!=null;
				}catch(Exception e){
					return false;
				}
			}
		}).map(new Function<Tuple2<String,Map<String,Object>>, LabeledPoint>() {
			public LabeledPoint call(Tuple2<String, Map<String, Object>> arg0) throws Exception {
				JSONObject jsonObject = new JSONObject(arg0._2);
				double[] array = new double[2];
				try{
					double used = Double.parseDouble(jsonObject.get("used").toString());
					double total = Double.parseDouble(jsonObject.get("total").toString());
					array[0] = used;
					array[1] = total;
					//上班日
//					array[2] = Double.parseDouble(isWorkingDay(jsonObject.get("date").toString()));
					//星期幾
//					array[2] = Double.parseDouble(getWeekOfDay(new JSONObject(arg0._2).get("date").toString()));
					//日
//					array[2] = Double.parseDouble(getDayOfMonth(new JSONObject(arg0._2).get("date").toString()));
					//月
//					array[2] = Double.parseDouble(getMonth(new JSONObject(arg0._2).get("date").toString()));
					double predict_day  = (total-used)/wr_average;
					
					return new LabeledPoint(predict_day, Vectors.dense(array));
				}catch(Exception e){
					return  new LabeledPoint(100,null);
				}
			}
		}).filter(new Function<LabeledPoint, Boolean>() {
			public Boolean call(LabeledPoint arg0) throws Exception {
				return arg0.features()!=null && arg0.label()!=0.0;
			}
		});
		
		
		Accumulator<Double> error =sc.accumulator(0.0);
		Accumulator<Double> mse =sc.accumulator(0.0);
		Accumulator<Double> mae =sc.accumulator(0.0);
		Accumulator<Double> rmse =sc.accumulator(0.0);
		Accumulator<Integer> time =sc.accumulator(0);
//		JavaRDD<LabeledPoint>[] splitRDD = jsonRDD.randomSplit(new double [] {0.7,0.3},11L);
		Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>>[] k_fold = MLUtils.kFold(jsonRDD.rdd(), 10, 11,jsonRDD.classTag());
		
		for(int i=0 ; i<1 ;i++){
			long start = System.currentTimeMillis();
			JavaRDD<LabeledPoint> trainRDD = k_fold[i]._1().toJavaRDD();
			JavaRDD<LabeledPoint> testRDD = k_fold[i]._2().toJavaRDD();
			trainRDD.cache();
			testRDD.cache();
			
			Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		    Integer numTrees = 10; // Use more in practice.
		    String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		    String impurity = "variance";
		    Integer maxDepth = 5;
		    Integer maxBins = 32;
		    Integer seed = 12345;
		    final RandomForestModel model = RandomForest.trainRegressor(trainRDD,
		      categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
		    
//			Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
//		    String impurity = "variance";
//		    Integer maxDepth =5;
//		    Integer maxBins = 32;
//		    final DecisionTreeModel model = DecisionTree.trainRegressor(trainRDD,
//		      categoricalFeaturesInfo, impurity, maxDepth, maxBins);
		
//			BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Regression");
//			boostingStrategy.setNumIterations(50); // Note: Use more iterations in practice.
//			boostingStrategy.getTreeStrategy().setMaxDepth(10);
//			Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
//		    boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);
//		    final GradientBoostedTreesModel model =GradientBoostedTrees.train(trainRDD, boostingStrategy);
			
//			int numIterations = 10000;
//		    double stepSize = 0.0000001;
//		    
//		    double miniBatchFraction = 1;
		    
//		    final LinearRegressionModel model =LinearRegressionWithSGD.train(JavaRDD.toRDD(trainRDD), numIterations, stepSize,miniBatchFraction);
//		    final RidgeRegressionModel model = RidgeRegressionWithSGD.train(testRDD.rdd(), numIterations, stepSize, miniBatchFraction);
		    
//		    final LassoModel model = LassoWithSGD.train(testRDD.rdd(), numIterations, stepSize, miniBatchFraction);
			
		  
			//訓練資料預測
			JavaRDD<Tuple2<Double, Double>> test_rdd = testRDD.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
				public Tuple2<Double, Double> call(LabeledPoint arg0) throws Exception {
					return new Tuple2<Double, Double>(model.predict(arg0.features()), arg0.label());
				}
			});
			
			
			JavaRDD<Tuple2<Double, Double>> check_rdd = test_rdd.filter(new Function<Tuple2<Double,Double>, Boolean>() {
				public Boolean call(Tuple2<Double, Double> arg0) throws Exception {
					return arg0._1().intValue() != arg0._2().intValue();
				}
			});
			
			for(Tuple2<Double, Double> v : check_rdd.take(50)){
				System.out.println(v);
			}
			
			double accuracy = check_rdd.count()/ (double)testRDD.count();
			System.out.println("error  " +  accuracy);
			
			double MSE = new JavaDoubleRDD(test_rdd.map(new Function<Tuple2<Double, Double>, Object>() {
				public Object call(Tuple2<Double, Double> pair) {
					return Math.pow(pair._1() - pair._2(), 2.0);
				}
			}).rdd()).mean();
			
			double MAE = new JavaDoubleRDD(test_rdd.map(new Function<Tuple2<Double, Double>, Object>() {
				public Object call(Tuple2<Double, Double> pair) {
					return Math.abs(pair._1() - pair._2());
				}
			}).rdd()).mean();
			
			
			System.out.println("training Mean Squared Error = " + MSE);
			System.out.println("training Root Mean Squared Error = " + Math.sqrt(MSE));
			System.out.println("training Mean Absolute Error = " + MAE);
			
			
			long end = System.currentTimeMillis();
			
			
			System.out.println((end-start)/1000+"  s");
			
			long v = ((end-start)/1000);
			
			error.add(accuracy);
			mse.add(MSE);
			rmse.add(Math.sqrt(MSE));
			mae.add(MAE);
			time.add((int)v);
			
			System.out.println(model.toDebugString());
			
		}
		
//		System.out.println("Average Error = " + error.value()/k_fold.length);
//		System.out.println("Average Mean Squared Error = " + mse.value()/k_fold.length);
//		System.out.println("Average Root Mean Squared Error = " +rmse.value()/k_fold.length);
//		System.out.println("Average Mean Absolute Error = " + mae.value()/k_fold.length);
//		System.out.println(time.value()/k_fold.length+" s");

	    sc.stop();
		
	}
	private static int getMonthNumber(String month){
		String [] arrStrings = {"Jan","Feb","Mar","Apr","Mar","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};
		for(int i=0 ;i <arrStrings.length ;i++){
			if(arrStrings[i].equals(month)){
				return i;
			}
		}
		return 0;
	}
	
	private static boolean isInTime(String dataDate) throws ParseException{
		Calendar calendar =  Calendar.getInstance();
		calendar.setTime(dateFormatUtc.parse(dataDate));
		int month = calendar.get(Calendar.MONTH);
		return (month<3)?true:false;
	}
	
	/**
	 * 是否是上班日 (0/1) (true/false)
	 */
	private static String isWorkingDay(String dataDate) throws ParseException{
		Calendar calendar =  Calendar.getInstance();
		calendar.setTime(dateFormatUtc.parse(dataDate));
		boolean boo = calendar.get(Calendar.DAY_OF_WEEK)!=Calendar.SATURDAY
				&& calendar.get(Calendar.DAY_OF_WEEK)!=Calendar.SUNDAY;
		System.out.println(boo);
		return (boo)?"0":"1";
	}
	/**
	 * @param dataDate
	 * @return 週日-週六 (1-7)
	 * @throws ParseException
	 */
	private static String getWeekOfDay(String dataDate) throws ParseException{
		Calendar calendar =  Calendar.getInstance();
		calendar.setTime(dateFormatUtc.parse(dataDate));
		return calendar.get(Calendar.DAY_OF_WEEK)+"";
	}
	
	/**
	 * 
	 * @param dataDate
	 * @return 月份 (1月-五月) (0-4)
	 * @throws ParseException
	 */
	private static String getMonth(String dataDate) throws ParseException{
		Calendar calendar =  Calendar.getInstance();
		calendar.setTime(dateFormatUtc.parse(dataDate));
		return calendar.get(Calendar.MONTH)+"";
	}
	/**
	 * 
	 * @param dataDate
	 * @return 日 (1-30)
	 * @throws ParseException
	 */
	private static String getDayOfMonth(String dataDate) throws ParseException{
		Calendar calendar =  Calendar.getInstance();
		calendar.setTime(dateFormatUtc.parse(dataDate));
		return calendar.get(Calendar.DATE)+"";
	}
	
	/**
	 * @param start 開始日期
	 * @param end 結束日期
	 * @return 回傳天數
	 */
	public static long getDayCount(String start, String end) {
		  long diff = -1;
		  try {
		    Date dateStart = dateFormatUtc.parse(start);
		    Date dateEnd = dateFormatUtc.parse(end);
		    diff = Math.round((dateEnd.getTime() - dateStart.getTime()) / (double) 86400000);
		  } catch (Exception e) {}
		  return diff;
	}
}
