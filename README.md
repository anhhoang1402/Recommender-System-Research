# **INSTRUCTIONS ON HOW TO RUN RECOMMENDATIONS.PY**

## **Overview of the program**

recommendations.py is a program that performs the following recommender system on the crtics and ML-100k dataset from grouplens.org:

1. User-Based collaborative filtering (UU-CF) with Pearson correlation and Euclidean Distance with similarity weighting (1,25,50) and similarity threshold as parameters 
2. Item-Based collaborative filtering (II-CF) with Pearson correlation and Euclidean Distance with similarity weighting (1,25, 50) and similarity threshold as parameters 
3. Matrix Factorization Stochastic Gradient Descent (MF-SGD) with number of factors, learning rate, and regularization rate as parameters
4. Matrix Factorization Alternating Least Squares (MF-ALS) with number of factors and regularization rate as parameters
5. Content-based recommender system Feature Enconding (FE)
6. Content-based recommender system Term Frequency - Inverse Document Frequency (TFIDF) with similarity threshold as parameter
7. Hybrid Recommender system (TFIDF and Item-Based): TFIDF-Pearson, TFIDF-Distance with weight parameter

## **The program provides the following functionalities**

1. Read data from the datasets:

* For UU-CF, II-CF, FE, TFIDF, and Hybrid: Run command R/r for critics dataset and command RML/rml for ML-100k datasets

* For MF-SGD and MF-ALS: Run command PD-R/pd-r for critics dataset and command pd-rml100 or PD-RML100 for ML-100k datasets

2. Provide statistics of the dataset:

* First read in data

* Run S(tats) command

3. Provide recommendations for users:

* First read in data

* For UU-CF: Run Simu command to produce a similarity matrix, click WP(Write Pearson) or WD (Write Distance) if it's the first time you run the program or you want
to produce a new similarity matrix with different parameter values; otherwise, run RP (Read Pearson) or RD (read distance) command to read in the similarity matrix, and run RECS to output recommendations

* For II-CF: Run Sim command to produce a similarity matrix, click WP(Write Pearson) or WD (Write Distance) if it's the first time you run the program or you want
to produce a new similarity matrix with different parameter values; otherwise, run RP (Read Pearson) or RD (Read Distance) command to read in the similarity matrix, and run RECS to output recommendations

* For TFIDF: Run tfidf command to set up consine similarity matrix, and run RECS to output recommendations

* For FE: Run FE command to set up FE similarity matrix, and run RECS to output recommendations

* For MF-ALS: run T command to set up the test/train data, then run MF-ALS to train dataset, and run RECS to output recommendations

* For MF-SGD: run T command to set up test/train data, then run MF-SGD to train dataset, and run RECS to output recommendations

* For Hybrid: Run H command to set up FE similarity matrix, and run RECS to output recommendations


4. Perform Leave One Cross Out Evaluation (LOOCV) to see the accuracy and coverage of recommender algorithms for UU-CF, II-CF, FE, TFIDF, and Hybrid :

* First read in data

* Run the command to set up the similarity matrix for the recommender method you want to perform LOOCV

* Run command LCVSIM

5. Perform Holdout Test and TrainCV to see the accuracy of recommender algorithms for MF-SGD and MF-ALS:

* First read in data

* Run the T command to set up train and test data

* Run MF-SGD command or MF-ALS depending on which algorithm you want to check

6. T-Test:

* Run T to output results for the T-test of our research paper
