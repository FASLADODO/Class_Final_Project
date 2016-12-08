1. Naive Bayes Model (Generative Method)

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

2. SVM Model (Discriminative Method)

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

3. KNN Model (Instance Based Method & Semi-supervised Dimensionality Reduction)

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

4. K means Model (Generative Method & Semi-supervised Dimensionality Reduction)

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