BEFORE YOU START
All models are ready to be run from "predict_labels.m", just comment in and out the code to the 
corresponding model. See more details below.

************************************************************************************************

1. Naive Bayes Model (Generative Method)

Overall approach: Assume that the words are independent. Since the data is sparse, we made the columns
indicators if a word was present in a tweet or not. Feature selection was done by mutual information
calculations to determine which features most described their corresponding labels. We used CV to determine 
how many of the top ranking features to include in the Naive Bayes classifier. 

Training Naive Bayes Model:

- open the "NB_script.m"
- run "Load the data" section
	this will load the training data as well as the variables used in the "predict_labels" function
- run "Naive Bayes PreProcess" section
	this will do the Mutual Information Feature Selection and save the selected feature index
- run "Naive Bayes Model generating" section
	this will generate the Naive Bayes model and save the model

Testing Naive Bayes Model:

- run "Naive Bayes prediction" section
	this section will call the "predict_labels" fuction.
	MAKE SURE TO uncomment the "Naive Bayes prediction" section within the "predict_labels.m" function

	The "predict_labels.m" function signature is just like what we submitted to leaderboard

Naive Bayes Model CV Accuracy: 0.8529

************************************************************************************************

2. SVM Model (Discriminative Method)

Overall approach: The SVM solver was chosen through trial and error, and we ended up using Sparse Reconstruction 
by Separable Approximation. Feature selection was done by mutual information calculations to determine which 
features most described their corresponding labels. We used CV to determine how many of the top ranking 
features to include in the SVM model. The threshold value that should be used to bucket the return values
from the model is determined at runtime by assuming the predictions should share the same distribution over 
the data set as the priors. 

Training SVM Model:

- open the "SVM_script.m"
- run "Load the data" section
	this will load the training data as well as the variables used in the "predict_labels" function
- run "SVM PreProcess" section
	this will do the Mutual Information Feature Selection and save the selected feature index
- run "SVM Model generating" section
	this will generate the SVM model and save the model

Testing SVM Model:

- run "SVM prediction" section
	this section will call the "predict_labels" fuction.
	MAKE SURE TO uncomment the "SVM prediction" section within the "predict_labels.m" function

	The "predict_labels.m" function signature is just like what we submitted to leaderboard

SVM Model CV Accuracy: 0.8084

************************************************************************************************

3. KNN Model (Instance Based Method) 

Overall approach: Chose the top 764 principal components that "described" 90% of the original 10,000 
features and clustered them into the 50 closest neighbors. The neighborhood size was determined 
through CV. 

Training KNN Model:

- open the "KNN_script.m"
- run "Load the data" section
	this will load the training data as well as the variables used in the "predict_labels" function
- run "KNN PreProcess" section
	this will do the PCA Feature Selection and save the feature coefficient
- run "KNN Model generating" section
	this will generate the KNN model and save the model

Testing KNN Model:

- run "KNN prediction" section
	this section will call the "predict_labels" fuction.
	MAKE SURE TO uncomment the "KNN prediction" section within the "predict_labels.m" function

	The "predict_labels.m" function signature is just like what we submitted to leaderboard

KNN Model CV Accuracy: 0.7273

************************************************************************************************

4. K means model with PCA feature reduction (Semi-supervised dimensionality reduction of the data)

Overall approach: Chose the top 764 principal components that "described" 90% of the original 10,000 
features. Created 500 clusters to try and group the tweets in categories. The cluster size was determined
by CV, although it had little affect. 

Training K means Model:

- open the "K_means_script.m"
- run "Load the data" section
	this will load the training data as well as the variables used in the "predict_labels" function
- run "K means PreProcess" section
	this will do the PCA Feature Selection and save the feature coefficient

Testing K means Model:

- run "K_means model building + Prediction" section
	this section will call the "predict_labels" fuction.
	MAKE SURE TO uncomment the "K_means prediction" section within the "predict_labels.m" function

	The "predict_labels.m" function signature is just like what we submitted to leaderboard

	K-Means/PCA Model CV Accuracy: 0.4447 :(