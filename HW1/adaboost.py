from feature import RectangleRegion, HaarFeature
from classifier import WeakClassifier
import utils
import numpy as np
import math
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle
from tqdm import tqdm


class Adaboost:
    def __init__(self, T):
        """
        Parameters:
          T: The number of weak classifiers which should be used.
        """
        self.T = 10
        self.alphas = []
        self.clfs = []

    def train(self, dataset):
        """
        Trains the Viola Jones classifier on a set of images.
          Parameters:
            dataset: A list of tuples. The first element is the numpy
              array with shape (m, n) representing the image. The second
              element is its classification (1 or 0).
        """
        print("Computing integral images")
        posNum, negNum = 0, 0
        iis, labels = [], []
        for i in range(len(dataset)):
            iis.append(utils.integralImage(dataset[i][0]))
            labels.append(dataset[i][1])
            if dataset[i][1] == 1:
                posNum += 1
            else:
                negNum += 1
        print("Building features")
        features = self.buildFeatures(iis[0].shape)
        print("Applying features to dataset")
        featureVals = self.applyFeatures(features, iis)
        print("Selecting best features")
        indices = (
            SelectPercentile(f_classif, percentile=10)
            .fit(featureVals.T, labels)
            .get_support(indices=True)
        )
        featureVals = featureVals[indices]
        features = features[indices]
        print("Selected %d potential features" % len(featureVals))

        print("Initialize weights")
        weights = np.zeros(len(dataset))
        for i in range(len(dataset)):
            if labels[i] == 1:
                weights[i] = 1.0 / (2 * posNum)
            else:
                weights[i] = 1.0 / (2 * negNum)
        for t in range(self.T):
            print("Run No. of Iteration: %d" % (t + 1))
            # Normalize weights
            weights = weights / np.sum(weights)
            # Compute error and select best classifiers
            clf, error = self.selectBest(featureVals, iis, labels, features, weights)
            # update weights
            accuracy = []
            for x, y in zip(iis, labels):
                correctness = abs(clf.classify(x) - y)
                accuracy.append(correctness)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0 / beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print(
                "Chose classifier: %s with accuracy: %f and alpha: %f"
                % (str(clf), (len(accuracy) - sum(accuracy)) / len(accuracy), alpha)
            )

    def buildFeatures(self, imageShape):
        """
        Builds the possible features given an image shape.
          Parameters:
            imageShape: A tuple of form (height, width).
          Returns:
            A numpy array of HaarFeature class.
        """
        height, width = imageShape
        features = []
        for w in range(1, width + 1):
            for h in range(1, height + 1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        # 2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i + w, j, w, h)
                        if i + 2 * w < width:  # Horizontally Adjacent
                            features.append(HaarFeature([right], [immediate]))

                        bottom = RectangleRegion(i, j + h, w, h)
                        if j + 2 * h < height:  # Vertically Adjacent
                            features.append(HaarFeature([immediate], [bottom]))

                        right_2 = RectangleRegion(i + 2 * w, j, w, h)
                        # 3 rectangle features
                        if i + 3 * w < width:  # Horizontally Adjacent
                            features.append(HaarFeature([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j + 2 * h, w, h)
                        if j + 3 * h < height:  # Vertically Adjacent
                            features.append(
                                HaarFeature([bottom], [bottom_2, immediate])
                            )

                        # 4 rectangle features
                        bottom_right = RectangleRegion(i + w, j + h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(
                                HaarFeature([right, bottom], [immediate, bottom_right])
                            )

                        j += 1
                    i += 1
        return np.array(features)

    def applyFeatures(self, features, iis):
        """
        Maps features onto the training dataset.
          Parameters:
            features: A numpy array of HaarFeature class.
            iis: A list of numpy array with shape (m, n) representing the integral images.
          Returns:
            featureVals: A numpy array of shape (len(features), len(dataset)).
              Each row represents the values of a single feature for each training sample.
        """
        featureVals = np.zeros((len(features), len(iis)))
        for j in tqdm(range(len(features))):
            for i in range(len(iis)):
                featureVals[j, i] = features[j].computeFeature(iis[i])
        return featureVals

    def selectBest(self, featureVals, iis, labels, features, weights):
        """
        Finds the appropriate weak classifier for each feature.
        Selects the best weak classifier for the given weights.
          Parameters:
            featureVals: A numpy array of shape (len(features), len(dataset)).
              Each row represents the values of a single feature for each training sample.
            iis: A list of numpy array with shape (m, n) representing the integral images.
            labels: A list of integer.
              The ith element is the classification of the ith training sample.
            features: A numpy array of HaarFeature class.
            weights: A numpy array with shape(len(dataset)).
              The ith element is the weight assigned to the ith training sample.
          Returns:
            bestClf: The best WeakClassifier Class
            bestError: The error of the best classifer
        """
        # Begin your code (Part 2)
        """
            This segment is responsible for finding the best weak classifier among
            a set of features for the given weights. It iterates over each feature and calculates
            the error associated with each one based on the feature values, labels, and weights. 
            The error is computed as the weighted sum of absolute differences between the predicted
            and actual labels. The feature that results in the lowest error is selected as the best
            weak classifier. Finally, it returns the best weak classifier and its corresponding error.
        """
        features_num = featureVals.shape[0]
        dataset_num = featureVals.shape[1]
        errors = np.zeros(features_num)
        
        for j in range(features_num):
            for i in range(dataset_num):
                errors[j] += weights[i] * abs((1 if featureVals[j][i] < 0 else 0) - labels[i])

        bestError = errors[0]
        bestClf = WeakClassifier(features[0])

        for i in range(1, features_num):
            if errors[i] < bestError:
                bestClf = WeakClassifier(features[i])
                bestError = errors[i]
        # End your code (Part 2)
        return bestClf, bestError
    
    # Begin your code (Part 6)
    """
        This implementation calculates the errors for all features using list comprehensions 
        and finds the minimum error using the min function. Then, it retrieves the corresponding 
        best classifier based on the index of the minimum error.
    """
    def my_selectBest(self, featureVals, iis, labels, features, weights):

        errors = []

        # Iterate over each row of featureVals, weights, and labels to compute errors
        for feature_row, weight, label in zip(featureVals, weights, labels):
            error = sum(weight * abs((1 if value < 0 else 0) - label) for value in feature_row)
            errors.append(error)

        # Find the index of the minimum error
        min_error_index = errors.index(min(errors))
        bestError = errors[min_error_index]
        bestClf = WeakClassifier(features[min_error_index])

        return bestClf, bestError
    # End your code (Part 6)

    def classify(self, image):
        """
        Classifies an image
          Parameters:
            image: A numpy array with shape (m, n). The shape (m, n) must be
              the same with the shape of training images.
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = utils.integralImage(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", "rb") as f:
            return pickle.load(f)
