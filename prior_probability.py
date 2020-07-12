import numpy as np

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        """
        count_true = np.count_nonzero(targets == 1)
        count_false = np.count_nonzero(targets == 0)
        self.most_common_class = True if count_true >= count_false else False
        return self.most_common_class

    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """

        # predict array same size as predict examples of all the most common class
        most_common_class = self.most_common_class
        num_rows = data.shape[0]
        predictions = np.zeros(num_rows, dtype = int) if most_common_class == False else np.ones(num_rows, dtype = int)
        return predictions

        

# feature = np.array([1,1,0,0,0,0])
# target = np.array([1,1,0,0,0,0])

# Prior = PriorProbability()
# print(Prior.most_common_class) 
# Prior.fit(feature, target)
# print(Prior.most_common_class)
# Prior.predict(feature)


# arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(arr2d.shape[1])
# print(arr2d)
# print(arr2d.ndim)
