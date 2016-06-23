package com.imac.openstack;

import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Date;
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
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.elasticsearch.spark.rdd.api.java.JavaEsSpark;
import org.json.simple.JSONObject;

import com.cloudera.sparkts.DateTimeIndex;
import com.cloudera.sparkts.DayFrequency;
import com.cloudera.sparkts.HourFrequency;
import com.cloudera.sparkts.MicrosecondFrequency;
import com.cloudera.sparkts.MinuteFrequency;
import com.cloudera.sparkts.api.java.DateTimeIndexFactory;
import com.cloudera.sparkts.api.java.JavaTimeSeries;
import com.cloudera.sparkts.api.java.JavaTimeSeriesFactory;
import com.cloudera.sparkts.api.java.JavaTimeSeriesRDD;
import com.cloudera.sparkts.api.java.JavaTimeSeriesRDDFactory;

import scala.Tuple2;

public class LogTS {
	private static org.apache.log4j.Logger LOGGER = org.apache.log4j.Logger.getLogger(LogTS.class);
	public static final Pattern nova_compute_log = Pattern
			.compile("([\\d*-]*) ([\\d*:]*.\\d*) (\\d*) (\\w*) ([\\w*\\.]*) \\[([\\w*-]*).*\\] (.*)(\\s*)");
	public static void main(String[] args) {
		
		SparkConf conf = new SparkConf();
		conf.set("es.index.auto.create", "true");
		conf.set("es.nodes", "10.26.1.9:9200");
		conf.set("es.resource", "openstack-log/ceilometer-agent-compute.log");
		conf.set("es.input.json", "true");
		conf.setAppName("TestStreaming");

		JavaSparkContext sc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(sc);
		
		JavaPairRDD<String, Map<String, Object>> esRDD = JavaEsSpark.esRDD(sc, "openstack-log/ceilometer-agent-compute.log");
		JavaRDD<Row> transactions = esRDD.map(new Function<Tuple2<String,Map<String,Object>>, Row>() {
			public Row call(Tuple2<String, Map<String, Object>> arg0) throws Exception {
				Map<String, Object> json_data =  arg0._2;
				JSONObject jsonObject = new JSONObject(json_data);
				String message = jsonObject.get("message").toString();
				String [] tokens = message.split(" ");
				String [] dates = tokens[0].split("-");
				String [] times = tokens[1].split(":");
				ZonedDateTime dt = ZonedDateTime.of(Integer.parseInt(dates[0]),
			            Integer.parseInt(dates[1]), Integer.parseInt(dates[2]), 
			            Integer.parseInt(times[0]), Integer.parseInt(times[1]), 0, 0,
			            ZoneId.systemDefault());
			    String host = jsonObject.get("host").toString();
		        double path = Double.parseDouble(jsonObject.get("@version").toString());
		        return RowFactory.create(Timestamp.from(dt.toInstant()), host, path);
			}
		});
		
//		esRDD.count();
		
		List<StructField> fields = new ArrayList();
	    fields.add(DataTypes.createStructField("timestamp", DataTypes.TimestampType, true));
	    fields.add(DataTypes.createStructField("host", DataTypes.StringType, true));
	    fields.add(DataTypes.createStructField("path", DataTypes.DoubleType, true));
	    StructType schema = DataTypes.createStructType(fields);
	    
	    schema.printTreeString();
	    
	    DataFrame dataframe = sqlContext.createDataFrame(transactions, schema);
	    
	    ZoneId zone = ZoneId.systemDefault();
	    DateTimeIndex dtIndex = DateTimeIndexFactory.uniformFromInterval(
	        ZonedDateTime.of(LocalDateTime.parse("2016-03-21T08:00:00"), zone),
	        ZonedDateTime.of(LocalDateTime.parse("2016-03-21T22:00:00"), zone),
	        new HourFrequency(1));
		
	    JavaTimeSeriesRDD<String> tickerTsrdd = JavaTimeSeriesRDDFactory.timeSeriesRDDFromObservations(
	            dtIndex, dataframe, "timestamp", "host", "path");
	    
	    tickerTsrdd.foreach(new VoidFunction<Tuple2<String,Vector>>() {
			public void call(Tuple2<String, Vector> arg0) throws Exception {
				System.out.println(arg0);
			}
		});

	    sc.stop();
		
	}

}
