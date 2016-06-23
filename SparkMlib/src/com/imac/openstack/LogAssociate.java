package com.imac.openstack;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import org.json.simple.JSONObject;

import scala.Tuple2;

public class LogAssociate {
	private static org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger(LogAssociate.class);
	private static SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
	public static final Pattern nova_compute_log2 = Pattern
			.compile("([\\d*-]*) ([\\d*:]*.\\d*) (\\d*) (\\w*) ([\\w*\\.]*) \\[([\\w*-]*) ([a-zA-Z0-9]*) ([a-zA-Z0-9]*).*-\\] \\[\\w*\\W\\s(.*)\\] (.*)(\\s*)");
	
	public static void main(String[] args) {
		SparkConf conf = new SparkConf();
		conf.set("es.index.auto.create", "true");
		conf.set("es.nodes", "10.26.1.9:9200");
		conf.set("es.resource", "openstack-log/nova-compute.log");
		conf.set("es.input.json", "true");
		conf.setAppName("TestStreaming");

		JavaSparkContext sc = new JavaSparkContext(conf);
		
		JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc, "openstack-log/nova-compute.log");
		JavaRDD<List<String>> transactions = esRDD.mapToPair(new PairFunction<Tuple2<String,Map<String,Object>>, String, String>() {
			public Tuple2<String, String> call(Tuple2<String, Map<String, Object>> arg0) throws Exception {
				Map<String, Object> json_data =  arg0._2;
				JSONObject jsonObject = new JSONObject(json_data);
				String message = jsonObject.get("message").toString();
				Matcher m = nova_compute_log2.matcher(message);
				if(m.find()){
					String key = m.group(8);
					String value ="";
					for(int i=5 ;i<m.groupCount() ; i++){
						if(i!=8){
							value+=m.group(i)+",";
						}
					}
					return new Tuple2<String, String>(key, value.substring(0, value.length()-1));
				}
				return new Tuple2<String, String>("","");
			}
		}).filter(new Function<Tuple2<String,String>, Boolean>() {
			public Boolean call(Tuple2<String, String> arg0) throws Exception {
				return !arg0._1.equals("");
			}
		}).filter(new Function<Tuple2<String,String>, Boolean>() {
			public Boolean call(Tuple2<String, String> arg0) throws Exception {
				return arg0._1.equals("6d1b52a49dbb48f180c0afa0ea92ef95");
			}
		}).values().map(new Function<String, List<String>>() {
			public List<String> call(String arg0) throws Exception {
				ArrayList<String> arrayList = new ArrayList<String>();
				for(String v : arg0.split(",")){
					arrayList.add(v);
				}
				return arrayList;
			}
		});
		
		
		FPGrowthModel<String> model = new FPGrowth()
	      .setMinSupport(0.1)
	      .setNumPartitions(10)
	      .run(transactions);

	    for (FPGrowth.FreqItemset<String> s: model.freqItemsets().toJavaRDD().collect()) {
	      LOGGER.warn("[" + s.javaItems()+ "], " + s.freq());
	    }
		

	    sc.stop();
		
	}
	
	

}
