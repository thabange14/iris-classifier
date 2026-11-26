package com.example.iris;

import smile.classification.KNN;
import tech.tablesaw.api.Table;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;

public class App {

    public static void main(String[] args) throws Exception {
        // 1. Load the data
        Table iris = Table.read().csv("data/iris.csv");
        System.out.println("Loaded " + iris.shape());
        System.out.println(iris.stringColumn("species").countByCategory());

        // 2. Turn the table into numbers the model can use
        String[] features = {"sepal_length", "sepal_width", "petal_length", "petal_width"};
        int n = iris.rowCount();

        double[][] x = new double[n][features.length];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < features.length; j++) {
                x[i][j] = iris.doubleColumn(features[j]).getDouble(i);
            }
        }

        // Species names -> numbers (setosa = 0, versicolor = 1, virginica = 2)
        List<String> classNames = new ArrayList<>(new TreeSet<>(iris.stringColumn("species").asSet()));
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = classNames.indexOf(iris.stringColumn("species").get(i));
        }

        // 3. Shuffle, then split 80% train / 20% test
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < n; i++) indices.add(i);
        Collections.shuffle(indices, new Random(42)); // fixed seed = same result every run

        int trainSize = (int) (n * 0.8);
        int testSize = n - trainSize;
        double[][] trainX = new double[trainSize][];
        int[] trainY = new int[trainSize];
        double[][] testX = new double[testSize][];
        int[] testY = new int[testSize];

        for (int i = 0; i < n; i++) {
            int idx = indices.get(i);
            if (i < trainSize) {
                trainX[i] = x[idx];
                trainY[i] = y[idx];
            } else {
                testX[i - trainSize] = x[idx];
                testY[i - trainSize] = y[idx];
            }
        }
        System.out.println("\nTraining on " + trainSize + " flowers, testing on " + testSize);

        // 4. Train the model
        KNN<double[]> model = KNN.fit(trainX, trainY, 5);

        // 5. Predict on the test flowers and measure accuracy
        int k = classNames.size();
        int[][] confusion = new int[k][k];
        int correct = 0;
        for (int i = 0; i < testSize; i++) {
            int predicted = model.predict(testX[i]);
            confusion[testY[i]][predicted]++;
            if (predicted == testY[i]) correct++;
        }
        System.out.printf("%nAccuracy: %.1f%% (%d of %d correct)%n",
                100.0 * correct / testSize, correct, testSize);

        // 6. Confusion matrix: rows = real species, columns = predicted species
        System.out.println("\nConfusion matrix (rows = actual, columns = predicted):");
        System.out.printf("%-12s", "");
        for (String name : classNames) System.out.printf("%-12s", name);
        System.out.println();
        for (int i = 0; i < k; i++) {
            System.out.printf("%-12s", classNames.get(i));
            for (int j = 0; j < k; j++) System.out.printf("%-12d", confusion[i][j]);
            System.out.println();
        }

        // 7. Try one brand new flower
        double[] mystery = {5.9, 3.0, 5.1, 1.8};
        System.out.println("\nNew flower " + java.util.Arrays.toString(mystery)
                + " is predicted to be: " + classNames.get(model.predict(mystery)));
    }
}