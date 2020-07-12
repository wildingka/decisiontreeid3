import numpy as np
import copy



class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        This class creates a DecisionTree using the Id3 algorithm. 

        Input: 
        attribute_names: list of strings of the attirbute names 

        Data Members: 
        attribute_names : stores the attribute_names
        tree: contains the nested tree structure created from tree class. 
            New trees are appended as a new branch is needed

        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def addnewtree(self, root, name, col, val):
        """
        Creates a new branch, which is itself a Tree, given a root to append it to. 
        Inputs: 
            root: Which previous node to append the branch to
            name: stores the attribute name that this node is split on 
            col: stores the attribute column that this node is split on
            val: Most common class, this is the classification for this branch if its a leaf
        """
        branch = Tree(attribute_name=name, attribute_index = col, value = val)
        root.branches.append(branch)
    
    def pickatt(self, features, targets, attributes):
        """
        Loops through remaining attributes to figure out which one has the highest information gain. 
        Inputs:
            features: NxK array of features and examples to loop through
            targets: Nx1 array of classifications for each example
            attributes: list of attribute names to loop through 
        """
        max_information_gain = -.1
        node_attribute_name = ''
        node_att_col = 0
        if (len(attributes) == 1):
            node_attribute_name = attributes[0]
            node_att_col = 0
        else:
            for y in range(len(attributes)):
                info_gain = information_gain(features,y,targets)
                if (info_gain > max_information_gain):
                    max_information_gain = info_gain
                    node_attribute_name = attributes[y]
                    node_att_col = y
        return node_attribute_name, node_att_col
    
    def splitvar(self, features, targets, column, value):
        rows = np.where(features[:,column] == value)
        newfeatures = features[rows[0],:]
        newfeatures = np.delete(newfeatures, column, 1)
        newtargets = targets[rows]
        return newfeatures, newtargets

    def most_common_class(self, targets):
        count_true = np.count_nonzero(targets == 1)
        count_false = np.count_nonzero(targets == 0)
        most_common_class = True if count_true >= count_false else False
        return most_common_class

    def ID3(self, features, targets, attributes, root, default=None):
        node_attribute_name, node_att_col = self.pickatt(features, targets, attributes)
        for x in range(2):
            newfeatures, newtargets = self.splitvar(features, targets, node_att_col, x)
            value = self.most_common_class(newtargets)
            
            if (newfeatures.shape[0] == 0) :
                return self
            if ((newfeatures.shape[1] == 0) or (len(np.unique(newtargets)) == 1)) :
                leaf_name = node_attribute_name + "leaf"
                self.addnewtree(root, leaf_name, node_att_col, value)
            else:
                self.addnewtree(root, node_attribute_name, node_att_col, x)
                newroot = root.branches[x]
                newattributes = attributes.copy()
                newattributes.remove(node_attribute_name)
                self.ID3(newfeatures, newtargets, newattributes, newroot, self.most_common_class(newtargets))

           
    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)
        if (self.tree == None):
            self.tree = Tree()
            if (len(self.attribute_names) == 0): # check there are attributes left
                return self
            if (len(np.unique(targets)) == 1):
                count_true = np.count_nonzero(targets == 1)
                count_false = np.count_nonzero(targets == 0)
                most_common_class = True if count_true >= count_false else False
                self.tree.attribute_name = most_common_class
                self.visualize()
                return self
        self.ID3(features, targets, self.attribute_names,self.tree, default = None)
    
    def findbranches(self, root, example):
        if (root.branches == []):
            prediction = root.value
            return prediction
        else:                 
            attribute_index = root.branches[0].attribute_index
            example_val = example[attribute_index]
            if (example_val > 1):
                example_val = 1
            newroot = root.branches[int(example_val)]
            newexample = np.delete(example, attribute_index)
            return self.findbranches(newroot, newexample)


    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        self._check_input(features)
        numexamples = features.shape[0]
        targets = np.zeros(numexamples)
        for row in range(numexamples):
            example = features[row,:]
            prediction = self.findbranches(self.tree, example)
            targets[row] = int(prediction)
        return targets   
        
def entropy(positives, negatives):
    """
    The difference in entropy is the characteristic used to measure information gain. 
    This function will calculate entropy for an array of outputs. This is a helper function for
    the information gain function below.
    Inputs: 
        positives: int, the number of examples classified as 1/True in your targets array
        negatives: int, the number of examples classified as 0/False in your targets array
    entropy = sum_t[-P_t*log_2(P_t)]
    
    """
    total = positives + negatives
    if (total == 0):
        entropy = 0
    else:
        pos_percent = float(positives/total)
        neg_percent = float(negatives/total)
        if (pos_percent > 0) & (neg_percent > 0): 
            entropy = -(pos_percent*np.log2(pos_percent)) - (neg_percent * np.log2(neg_percent))
        elif (pos_percent > 0) & (neg_percent == 0):
            entropy = -(pos_percent*np.log2(pos_percent))
        else: 
            entropy = -(neg_percent * np.log2(neg_percent))
    return entropy

def information_gain(features, attribute_index, targets):
    """
    Calculates the information gain for each possible attribute. The attribute with the highest 
    information gain will be chosen to be the next split in the decision tree. This algorithm 
    uses entropy to calculate information gain. This splits each attribute on every value it takes 
    on. Thus is targeted to binary/categorical variables rather than continuous.
    Inputs:
        features: NxK array of data of examples
        attribute_index: int, columnn number of the attribute to calculate informataion gain for
        targets: Nx1 array of classifications for examples
    Outputs:
        information_gain: float, information gained for that variable if it were to be split on
    """
    attribute = features[:,attribute_index]
    levels = len(np.unique(attribute))
    total = len(attribute)
    entropy_after = 0
    if (total == 0):
        entropy_after = 0
    else:
        for x in range(levels): # the number of unique attributes
            total_level = np.count_nonzero(attribute == x) #count the total number of obs with that instance of att
            positives = np.count_nonzero((attribute == x) & (targets == 1)) #number of this instance with pos outcome
            negatives = np.count_nonzero((attribute == x) & (targets == 0)) #number of this instance with neg outcome
            assert total_level == positives + negatives #check our pos and negs add to toal of this att
            part_entropy = float(total_level/total)*(entropy(positives, negatives)) #prob of this instance of att*entropy(att)
            entropy_after = entropy_after + part_entropy #add this instance's entropy to total attribut
    entropy_before = entropy(np.count_nonzero(targets == 1),np.count_nonzero(targets == 0))
    information_gain = entropy_before-entropy_after
    return information_gain

