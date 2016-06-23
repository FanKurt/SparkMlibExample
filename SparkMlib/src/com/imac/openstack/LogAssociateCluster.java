package com.imac.openstack;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import org.json.simple.JSONObject;

import breeze.linalg.trace;

import scala.Tuple2;

public class LogAssociateCluster {
	private static org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger(LogAssociateCluster.class);
	private static SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
	public static final Pattern nova_compute_log2 = Pattern
			.compile("([\\d*-]*) ([\\d*:]*.\\d*) (\\d*) (\\w*) ([\\w*\\.]*) \\[([\\w*-]*) ([a-zA-Z0-9]*) ([a-zA-Z0-9]*).*-\\] \\[\\w*\\W\\s(.*)\\] (.*)(\\s*)");
	static ArrayList<String> cluster0 = new ArrayList<>();
	static ArrayList<String> cluster1 = new ArrayList<>();
	static ArrayList<String> cluster2 = new ArrayList<>();
	public static void main(String[] args) {
		SparkConf conf = new SparkConf();
		conf.set("es.index.auto.create", "true");
		conf.set("es.nodes", "10.26.1.9:9200");
		conf.set("es.resource", "openstack-log/nova-compute.log");
		conf.set("es.input.json", "true");
		conf.setAppName("TestStreaming");

		JavaSparkContext sc = new JavaSparkContext(conf);
		
		JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc, "openstack-log/nova-compute.log");
		JavaRDD<List<String>> transactions = esRDD.map(new Function<Tuple2<String,Map<String,Object>>, List<String>>() {
			public List<String> call(Tuple2<String, Map<String, Object>> arg0) throws Exception {
				Map<String, Object> json_data =  arg0._2;
				JSONObject jsonObject = new JSONObject(json_data);
				String message = jsonObject.get("message").toString();
				Matcher m = nova_compute_log2.matcher(message);
				ArrayList<String>arrayList = new ArrayList<String>();
				if(m.find()){
					for(int i=1 ;i<m.groupCount();i++){
						arrayList.add(m.group(i));
					}
				}
//				for(String v : message.split(" ")){
//					arrayList.add(v);
//				}
				return arrayList;
			}
		}).filter(new Function<List<String>, Boolean>() {
			public Boolean call(List<String> arg0) throws Exception {
				return arg0.size()>0;
			}
		});
		
		
		JavaRDD<ArrayList<Tuple2<Integer, String>>> catFeatureRDD=transactions.map(new Function<List<String>, ArrayList<Tuple2<Integer, String>>>() {
			public ArrayList<Tuple2<Integer, String>> call(List<String> arg0) throws Exception {
				return parseCatFeatures(arg0);
			}
		});
		final Map<Tuple2<Integer, String>, Long> catMap = catFeatureRDD.flatMap(new FlatMapFunction<ArrayList<Tuple2<Integer,String>>, Tuple2<Integer, String>>() {
			public Iterable<Tuple2<Integer, String>> call(ArrayList<Tuple2<Integer, String>> arg0) throws Exception {
				return arg0;
			}
		}).distinct().zipWithIndex().collectAsMap();
		
		JavaRDD<Vector> points = transactions.map(new Function<List<String>, Vector>() {
			public Vector call(List<String> arg0) throws Exception {
				ArrayList<Tuple2<Integer, String>> cat_data = parseCatFeatures(arg0);
				double[] doubleArray = new double [cat_data.size()];
				
				for(int i=0; i<cat_data.size(); i++){
					Tuple2<Integer, String> value = cat_data.get(i);
					if(catMap.containsKey(value)){
						doubleArray[i]=(double)catMap.get(value);
					}else{
						doubleArray[i]=0.0;
					}
				}
				return Vectors.dense(doubleArray);
			}
		});
		final KMeansModel model = KMeans.train(points.rdd(), 3, 10,
				3, KMeans.K_MEANS_PARALLEL());
		

		
		for(List<String> arg0 : transactions.collect()){
			ArrayList<Tuple2<Integer, String>> cat_data = parseCatFeatures(arg0);
			double[] doubleArray = new double [cat_data.size()];
			
			for(int i=0; i<cat_data.size(); i++){
				Tuple2<Integer, String> value = cat_data.get(i);
				if(catMap.containsKey(value)){
					doubleArray[i]=(double)catMap.get(value);
				}else{
					doubleArray[i]=0.0;
				}
			}
			
			int clusterNum = model.predict(Vectors.dense(doubleArray));
			if(clusterNum == 0){
				cluster0.add(arg0.toString());
			}else if (clusterNum==1){
				cluster1.add(arg0.toString());
			}else{
				cluster2.add(arg0.toString());
			}
		}
		

		JavaRDD<String> cluster0_rdd = sc.parallelize(cluster0);
		JavaRDD<String> cluster1_rdd = sc.parallelize(cluster1);
		JavaRDD<String> cluster2_rdd = sc.parallelize(cluster2);
		
		JavaRDD<List<String>> rdd =  cluster2_rdd.map(new Function<String, List<String>>() {
			public List<String> call(String arg0) throws Exception {
				ArrayList<String> arrayList = new ArrayList<>();
				String [] tokens = arg0.trim().split(",");
				for(String value : tokens){
					arrayList.add(value);
				}
				return arrayList;
			}
		});
		
		FPGrowthModel<String> fp = new FPGrowth()
	      .setMinSupport(0.1)
	      .setNumPartitions(10)
	      .run(rdd);

	    for (FPGrowth.FreqItemset<String> s: fp.freqItemsets().toJavaRDD().collect()) {
	      LOGGER.warn("[" + s.javaItems()+ "], " + s.freq());
	    }
	    sc.stop();
		
	}
	//One-Hot Encoding
	public static ArrayList<Tuple2<Integer, String>> parseCatFeatures(List<String> list){
		ArrayList<Tuple2<Integer, String>> arrayList = new ArrayList<>();
		for(int i=0 ; i <list.size();i++){
			arrayList.add(new Tuple2<Integer, String>(i, list.get(i)));
		}
		return arrayList;
	}
}
