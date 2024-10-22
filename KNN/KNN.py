import numpy as np
from collections import Counter

# Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    
    def __init__(self, k=3):
        self.k = k
    
    # Store the training data
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    # Predict for a single instance  
    def predict_one_instance(self, x):
        
        # Calculate distances between x and all training instances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the indices of the k-nearest instances 
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k-nearest instances
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return a list of tuples where each tuple contains a label and its frequency 
        most_common = Counter(k_nearest_labels).most_common()
        
        return most_common[0][0]

    
    # Predict for all instances
    def predict(self, X_test):
        
        predictions = [self.predict_one_instance(x) for x in X_test]
        
        return np.array(predictions)
    


