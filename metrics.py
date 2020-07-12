import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    true_negatives = np.count_nonzero((actual == 0) & (predictions == 0)) #Observation is negative, and is predicted to be negative. 
    false_positives = np.count_nonzero((actual == 0) & (predictions == 1)) #Observation is negative, but is predicted positive.
    false_negatives = np.count_nonzero((actual == 1) & (predictions == 0)) #Observation is positive, but is predicted negative.
    true_positives = np.count_nonzero((actual == 1) & (predictions == 1)) #Observation is positive, and is predicted to be positive.
    print(true_negatives, false_negatives, true_positives, false_positives, actual.shape[0])
    assert true_negatives + false_positives + true_positives + false_negatives == actual.shape[0]
    conf_matrix = np.array([[true_negatives,false_positives],[false_negatives,true_positives]])
    return conf_matrix

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    conf_matrix = confusion_matrix(actual, predictions)
    # confusion_matrix = np.array([[true_negatives,false_positives],[false_negatives,true_positives]])
    TP = conf_matrix[1,1]
    TN = conf_matrix[0,0]
    FP = conf_matrix[0,1]
    FN = conf_matrix[1,0]
    acc = float((TP + TN)/(TP+TN+FP+FN))
    return acc

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    #recall = TP /(TP+FN)
    #precision = TP / (TP+FP)
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    conf_matrix = confusion_matrix(actual, predictions)
    # confusion_matrix = np.array([[true_negatives,false_positives],[false_negatives,true_positives]])
    TP = conf_matrix[1,1]
    # TN = conf_matrix[0,0]
    FP = conf_matrix[0,1]
    FN = conf_matrix[1,0]
    rec = float(TP/(TP+FN))
    prec = float(TP/(TP+FP))
    return prec, rec

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    #F-measure = (2*recall*precision)/(recall+precision)
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)
    f1 = float(2*recall*precision)/(recall+precision)
    return f1
