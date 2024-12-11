The purpose of this project is to investigate the optimal visualization and classification of navigation strategies using machine learning (ML) method.
The objective of conducting this research is to predict the methods that the participants will emply to travel to the location which they are not familiar,
where the dataset is provided by Dr. Jimmy Zhong which the experiment is carry out in the National University Singapore (NUS). The tataics fit into three main 
categories which are procedural route, allocentric survey, and spatial update. The objective of the current study is to:
1. Addressing shortcomings in traditional and other machine learning models' factor score computation, classification, and data visualization.
2. To create comprehensible strategy classification, strategy prefernece and data visualization, indices using machine learning.
3. To facilitate the quick and precise categorization of freshly reported navigation strategy ratings through the utilization of machine learning techniques,
   eliminating the necessity of gathering additional data from an extensive participant pool.

For the LDA, KPCA, PCA model, the coordinate of training data and testing data is saved inside a csv file in order for further investigation used.
The test-size of 0.2 will be used in order to get a more accuracte result while making prediction for the test data.
The LDA models shows it had the best result for classification this dataset. therefore, the coordinate for the train and test data had been save in csv file for investigation.
The SHAP model will be using to generate a violin summary plot which summarizing up the effect of the most important NSQ items on sketchmap classification.
