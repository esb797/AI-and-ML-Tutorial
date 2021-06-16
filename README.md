# What is AI?
The Artificial Intelligence (AI) is a big umbrella that covers all classical programming, machine learning, and deep learning. Two famous cases are a Jeopardy quiz show and a game of Go. In the case of a Jeopardy quiz show, the IBM’s Q&A system Watson defeated the two Jeopardy champions. In the case of a game Go, the AlphaGo won 4 out of 5 games of Go in a match with a Go champion in 2016 and again won a three-game match with the No. 1 ranking player in 2017.

The machine learning is a field where a machine learns the rules of the process or system and it typically categorized as either supervised or unsupervised. The deep leaning is called deep since it consists of multiple layers of computations. The natural language processing or computer vision utilizes the deep learning method.

Algorithm vs. Model:

An algorithm is a method or a procedure to solve a problem. Whereas a model is a computation or a formula that takes some values as input and produces some value(s) as output.

## What is Machine Learning?

Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a subset of [artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence). Machine learning algorithms build a [mathematical model](https://en.wikipedia.org/wiki/Mathematical_model) based on sample data, known as ["training data"](https://en.wikipedia.org/wiki/Training_data), in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as [email filtering](https://en.wikipedia.org/wiki/Email_filtering) and [computer vision](https://en.wikipedia.org/wiki/Computer_vision), where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks (source: [Wikipedia](https://en.wikipedia.org/wiki/Machine_learning)).

When a rule or logic of the process or system is known that rule can be written as a program (algorithm) to automate the process or system without repeating it over. However, when a rule or logic or pattern is not known, using the existing data (typically, the historical data), one can learn the rules through the machine learning (ML) algorithms. Therefore, the easy way to see distinguish the ML is to see if the rules are known or not.

### Supervised Machine Learning

In case of the supervised machine learning, the data has a target to be predicted. First, the data is split into three parts: training, validating, and testing. From these split data sets, the training and validating sets are fed into the algorithm to create a model. Once model is built, the test set is passed to the model to identify the model quality such as accuracy or over-fitting. Since the model never saw the test dataset, there is no cheating when building the model; thus, this model quality checking phase simulates how the model would perform with the new data. Once the model quality is satisfactory, the model can be deployed to predict on the new incoming data.

### Unsupervised Machine Learning

An unsupervised machine learning learns from the unlabeled data and provides unlabeled groups or aggregated data. It learns the features or patterns of the data.

# Tutorials

## Classification
A classification model can be categorized into two types: binary and multi-class. A binary classification has only two classes denoted as either 0 or 1. Whereas, a multi-class (multinomial) classification has more than 2 classes (target values). A multi-class classification model typically works either one vs. rest or one vs. one. In the case of one vs. rest, in each stage, we consider one class at a time to train a binary classification with the considered class as positive and all other classes as negatives

- Binary: 0 or 1. For example, we need to predict whether a picture is a dog or not-a-dog.
- Multi-class: more than 2 classes (target values). For example, we have four kinds of animal pictures (dog, cat, mouse, and pig) and we need to predict if a picture is a dog or a cat or a mouse or a pig
   - One vs. Rest: In each step, we consider one class at a time. Using the same example of four animals, at first, we consider a dog as a positive and all other animals as negative. So, the first stage is predicting if a picture is a dog or not-a-dog. And we repeat for the rest of the three animals.
   - One vs. One: A training is conducted at once by considering all animal pictures. 

Terminology:

- true positive (TP): the actual label is positive, and the prediction is also positive.
- true negative (TN): the actual label is negative, and the prediction is also negative.
- false positive (FP): the actual label is negative, and the prediction is positive.
- false negative (FN): the actual label is positive, and the prediction is negative.
- true positive rate (TPR) or recall:
  <img src="https://render.githubusercontent.com/render/math?math=TPR = \frac{TP}{TP %2B FN}">
- positive predictive value (PPV) or precision:
  <img src="https://render.githubusercontent.com/render/math?math=PPV = \frac{TP}{TP %2B FP}">

Confusion matrix: a table layout to visualize the model performance.

Classification model quality metrics:
- Accuracy: ratio of correct predictions to all predictions
- AUC: area under the curve (true positive rate vs. false positive rate)
- F1 score: harmonic mean of precision and recall
   <img src="https://render.githubusercontent.com/render/math?math=F_{1} = \frac{2TP}{2TP %2B FP %2B FN}}">

### Classification Tutorial

Level: Beginner

Time: approximately 1 hour for each tutorial

- [Logistic Regression](https://careerfoundry.com/en/blog/data-analytics/what-is-logistic-regression/): [tutorial](https://nickmccullum.com/python-machine-learning/logistic-regression-python/)
- [Support Vector Machine (SVM)](https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_classification_algorithms_support_vector_machine.htm): [more tutorial](https://towardsdatascience.com/the-complete-guide-to-support-vector-machine-svm-f1a820d8af0b)
- [Decision Tree](https://www.logic2020.com/insight/tactical/decision-tree-classifier-overview):[ tutorial](https://www.datacamp.com/community/tutorials/decision-tree-classification-python)

## Recommendation Engine (System)

A recommendation engine (system) can be categorized into two types: collaborative or content based. The collaborative approach assumes that people who agreed in the past will agree in the future (the recommendation is based on the past behavior). Whereas the content-based approach uses a description of the item and a profile of the user’s preferences.

### Recommendation Engine Tutorials

Level: Beginner

Time: approximately 1 hour for each tutorial

- [Recommender Systems with Python — Part I: Content-Based Filtering](https://heartbeat.fritz.ai/recommender-systems-with-python-part-i-content-based-filtering-5df4940bd831)
- [Recommender Systems with Python— Part II: Collaborative Filtering (K-Nearest Neighbors Algorithm)](https://heartbeat.fritz.ai/recommender-systems-with-python-part-ii-collaborative-filtering-k-nearest-neighbors-algorithm-c8dcd5fd89b2)
- [Recommender Systems with Python — Part III: Collaborative Filtering (Singular Value Decomposition)](https://heartbeat.fritz.ai/recommender-systems-with-python-part-iii-collaborative-filtering-singular-value-decomposition-5b5dcb3f242b)

## Anomaly Detection

Anomaly detection is the identification of rare items, events, or observations that are significantly different from most of the data.

With the known anomalies (“labels”) in the data, we can build a classification model (supervised machine learning). Once model is trained and built, the model can detect an anomaly in real-time.

If the anomaly is not known or too rare for the supervised machine learning, we can use the unsupervised machine learning algorithms. A snapshot of data is needed to detect anomalies, so this is not for a real-time anomaly detection.

### Anomaly Detection Tutorials

Level: Beginner

Time: approximately 1 hour for each tutorial

- [Anomaly Detection for Dummies](https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1)
- [Comparing anomaly detection algorithms for outlier detection on toy dataset (Scikit-Learn)](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html)
- [Fraud Detection](https://towardsdatascience.com/fraud-detection-unsupervised-anomaly-detection-df43d81fce67)
- [Time Series Forecasting with Prophet in Python](https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/)

## Topic Model
A topic model is a type of statistical model to discover “topics” or “themes” within a collection of documents. The main idea is to group similar documents into one “topic” or “theme”.

### Topic Model Tutorials

Level: Intermediate

Time: approximately 1 hour for each tutorial

- https://nlpforhackers.io/topic-modeling/
- https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

## Azure Machine Learning Tutorials

Level: Intermediate

Time: approximately 1 hour for each tutorial

- Diabetes disease progression prediction:
>- [Set up the workspace and dev environment](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup)
>- [Train the model](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-train)
- Image classification (MNIST data):

>- [Train a model](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-train-models-with-aml)
>- [Deploy a model](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-deploy-models-with-aml)

- Regression with Automated ML (NYC Taxi data)
>- [Auto-train an ML model](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-auto-train-models)

- Azure Machine Learning Examples
>- [GitHub Repo](https://github.com/Azure/azureml-examples)

## Deep Learning
Deep learning is a type of machine learning (within the machine learning field) and it is based on artificial neural networks.

Level: Advanced

Time: varies for each topic

- [PyTorch Tutorial](https://pytorch.org/tutorials/)
- [Tensorflow Tutorial](https://www.tensorflow.org/tutorials/)
- [Deep Learning on Azure Databricks](https://towardsdatascience.com/how-to-create-your-own-deep-learning-project-in-azure-509660d8297)

# Videos
- [Data Science Dojo](https://tutorials.datasciencedojo.com/video-series/data-science-in-minutes/)

# What questions to ask to solve a business problem
- Do you have text to be analyzed? -> Natural Language Process
- Do you have labels/classes? -> Supervised machine learning
- Do you need to identify trends among vast of data? -> Unsupervised machine learning

[Azure ML Cheat Sheet](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-cheat-sheet)
