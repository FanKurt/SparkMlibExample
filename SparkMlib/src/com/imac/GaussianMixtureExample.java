package com.imac;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;

public class GaussianMixtureExample {
  public static void main(String[] args) {
    SparkConf conf = new SparkConf().setAppName("GaussianMixture Example");
    JavaSparkContext sc = new JavaSparkContext(conf);

    // Load and parse data
    String input = args[0];
    String output = args[1];
    JavaRDD<String> data = sc.textFile(input);
    JavaRDD<Vector> parsedData = data.map(
      new Function<String, Vector>() {
        public Vector call(String s) {
          String[] sarray = s.trim().split(" ");
          double[] values = new double[sarray.length];
          for (int i = 0; i < sarray.length; i++)
            values[i] = Double.parseDouble(sarray[i]);
          return Vectors.dense(values);
        }
      }
    );
    parsedData.cache();

    // Cluster the data into two classes using GaussianMixture
    GaussianMixtureModel gmm = new GaussianMixture().setK(2).run(parsedData.rdd());

    // Save and load GaussianMixtureModel
    gmm.save(sc.sc(), output);
    GaussianMixtureModel sameModel = GaussianMixtureModel.load(sc.sc(), output);
    // Output the parameters of the mixture model
    for(int j=0; j<gmm.k(); j++) {
        System.out.printf("weight=%f\nmu=%s\nsigma=\n%s\n",
            gmm.weights()[j], gmm.gaussians()[j].mu(), gmm.gaussians()[j].sigma());
    }
  }
}