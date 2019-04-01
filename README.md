# discrete_consistency_info
a little library containing a single class to help me with evaluating discrete data consistency.

mushrooms.csv is a dataset from kaggle
https://www.kaggle.com/uciml/mushroom-classification

Instructions:
Begin by creating an instance of the ConsistencyInfo() class:
  ci = ConsistencyInfo(x_vectors, x_colNames, y_vector, *tol=15)

x_vectors(numpy array): a transposed matrix of the x columns.

x_colNames(numpy array/list): an array/list of the x column names.

y_vector(numpy array): a n-dimensional vector with n y elements.

tol(int/float)(optional, default=15): a tolerance value for high number of unique attributes in features. The higher the number, the         higher the tolerance, as long as the number is above 1. If 0 < tol < 1, then the tolerance for highly unique attributes will start         getting bigger as tol approaches 0.



attributes of the ConsistencyInfo class:

  .sorted_pairs: a list of features in the form [[consistency, feature name, original index in x_vectors], ...]

  .top_feature: the most consistent feature in the form [consistency, feature name, original index in x_vectors]
