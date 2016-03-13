Four different methods are checked :- 
 [1] Linear Regression
 [2] Ridge Regression
 [3] Lasso
 [4] SVR

For all these methods cross validation is performed on the training data (available as bikeDataTrainingUpload.csv). 
The split considered for validation phase was : 80% for the training set and 20 % for the validation set. 
Since the dataset consisted of both discrete and continuous variables I have used 'one-hot 'encoding scheme for the 
discrete variables. Then the models(each for Linear Regression, Ridge Regression, Lasso, SVR) are trained using the training data (80%) and tested using the validation set(20%).

Performance:

SVR gave the lowest RMSE error among all the tested models. The parameters for SVR especially C and epsilon are chosen after repeated grid search.  So the optimal parameters for SVR are C:61900 epsilon=82. Kernel used was rbf. 

The parameters thus learned are used to train the entire training dataset in clf_tot. The final prediction for the cnt variables
was done using the test dataset provided in the file (TestX.csv).


I have run the code on Spyder IDE . So I didnot require commands for running the code. The commands for running the code :
(filename-->'Assignment1_ML.py')

The program file :Assignment1_ML.py
Output file: 143070026.csv


 

