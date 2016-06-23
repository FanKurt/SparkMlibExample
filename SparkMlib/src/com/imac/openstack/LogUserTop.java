package com.imac.openstack;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
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

public class LogUserTop {
	private static org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger(LogUserTop.class);
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
		JavaPairRDD<String, String> transactions = esRDD.mapToPair(new PairFunction<Tuple2<String,Map<String,Object>>, String, String>() {
			public Tuple2<String, String> call(Tuple2<String, Map<String, Object>> arg0) throws Exception {
				Map<String, Object> json_data =  arg0._2;
				JSONObject jsonObject = new JSONObject(json_data);
				String message = jsonObject.get("message").toString();
				Matcher m = nova_compute_log2.matcher(message);
				if(m.find()){
					String key = m.group(7);
					String value ="";
					return new Tuple2<String, String>(key,m.group(10));
				}
				return new Tuple2<String, String>("","");
			}
		}).filter(new Function<Tuple2<String,String>, Boolean>() {
			public Boolean call(Tuple2<String, String> arg0) throws Exception {
				return !arg0._1.equals("");
			}
		}).filter(new Function<Tuple2<String,String>, Boolean>() {
			public Boolean call(Tuple2<String, String> arg0) throws Exception {
			System.out.println(UserID.isContainUserToken(arg0._1));
			return UserID.isContainUserToken(arg0._1) && arg0._2.contains("Attempting claim");
			}
		}).groupByKey().mapToPair(new PairFunction<Tuple2<String,Iterable<String>>, String, String>() {
			public Tuple2<String, String> call(Tuple2<String, Iterable<String>> arg0) throws Exception {
				Iterator<String> iterator =arg0._2.iterator();
				HashMap<String, String> userID = UserID.getUserMap();
				int memory = 0,disk = 0;
				while(iterator.hasNext()){
					String value = iterator.next();
					String [] tokens = value.split(",");
					String [] split1 = tokens[0].split(" ");
					String [] split2 = tokens[1].split(" ");
					memory+=Integer.parseInt(split1[3]);
					disk+=Integer.parseInt(split2[2]);
				}
				return new Tuple2<String, String>("memory "+memory+", disk "+disk,userID.get(arg0._1));
			}
		}).sortByKey();
		
		for(Tuple2<String, String> v : transactions.collect()){
			LOGGER.warn(v);
		}	
	    sc.stop();
		
	}
	
	static class UserID{
		private static String user_id= "22f96f3ce57245b2a2318cf5fa679013 | a0936196693,"
							+"3bed7813ca904235bdef727fc38cdea9 | murano,"
							+"43890b74c50a43ddb9ff0e16172a5d54 | nova,"
							+"4946cb52afd44015b35efb94da260cb3 | swift,"
							+"4bd3c1b308a44c639a346a0b75260e4b | MaxJiang,"
							+"5cfd9ea86f5c41eabd9aa5d543d05be1 | k753357,"
							+"6b0ecf1c6e3e4d50b8f40bdcbc050706 | heat,"
							+"79bf70e860474dffbb2b77df1a19ce99 | aionshun,"
							+"80c6574faeeb441cbf5ebf482113ce20 | yangbx,"
							+"8657e8ede2b44c67801158a8bd9fba89 | cijie.l,"
							+"8dfcbc39626043d98021e4f26fddf5bc | cinder,"
							+"94e67041569e412286fd4e9c6052398b | imac-cloud,"
							+"a2bab117debc4976b5779fe938057143 | webberkuo,"
							+"a2e7c7b695a34b628e4db1eb6b8eb95c | bokai,"
							+"a3b6380da96c489aa164336e3423f6f3 | kyle.bai,"
							+"a900a6ce827c43659dad8cd5bac01536 | jigsawye,"
							+"b264b365cf3d4b6cafb051794d9d54c2 | chaoen,"
							+"bf1be934d5aa4a8bb3e68940b59e4c0c | ellis.wu,"
							+"c071d94ebe8f48ec99060c74d1b8c9d9 | 1401k012,"
							+"c64063dc09ba446f9c9216bbb616addf | glance,"
							+"cabbefcee7224ee9a251251a633b36ad | magnum,"
							+"cc3bea766b6a42d29fba93ea71686092 | kiss5891,"
							+"cf5801f668774781b83825cd2069afb2 | neutron,"
							+"e3ee6724b17e4f7ea7abc41fb3502bb8 | ceilometer,"
							+"faab77235cec4ac4a9e771ee8f4508db | jacky05308";
					
		private static HashMap<String, String> getUserMap(){
			String [] tokens = user_id.split(",");
			HashMap<String, String> mHashMap = new HashMap<>();
			for(String value : tokens){
				String [] split = value.trim().split("\\|");
				mHashMap.put(split[0].trim(), split[1].trim());
			}
			return mHashMap;
		}
	   private static Boolean isContainUserToken(String str){
		  for(String v : getUserMap().keySet()){
			  if(str.contains(v)){
				  return true;
			  }
		  }
		  return false;
	  }	
		
	}

}
