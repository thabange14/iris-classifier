# Iris Classifier (Java)

A small machine learning project in Java that classifies iris flowers into three species (setosa, versicolor, virginica) from their petal and sepal measurements.

## Tech stack

- **Java 21+** and **Maven**
- **[Tablesaw](https://github.com/jtablesaw/tablesaw)** for loading and exploring data (like pandas)
- **[Smile](https://haifengl.github.io/)** for machine learning (like scikit-learn)

## What it does

1. Loads the Iris dataset (150 flowers, 4 measurements each) from `data/iris.csv`
2. Splits it into 80% training and 20% test data
3. Trains a K-Nearest Neighbors classifier (k = 5)
4. Evaluates it with accuracy and a confusion matrix
5. Predicts the species of a new flower

## Results

- **Accuracy:** 96.7% on the 30 test flowers (replace XX with your number)

## How to run

```bash
git clone https://github.com/thabange14/iris-classifier.git
cd iris-classifier
mvn -q compile exec:java
```

You need Java 21 or newer and Maven installed.

## Project structure

```
iris-classifier/
├── data/iris.csv                         # dataset
├── pom.xml                               # Maven dependencies
└── src/main/java/com/example/iris/App.java   # main program
```

## Dataset

The Iris dataset was introduced by Ronald Fisher in 1936 and is a classic for learning classification.

## License

MIT