from utilities import Utilities
from signature_feature_extraction import SignatureFeatureExtraction
import config


class SignatureTraining:
    
    def training_genuine_with_soft_dtw_without_gradient(location, trainingSize):
        print("Training on Genuine Signatures...")
        featureList = []  # define an empty list to hold features 
        for i in range(1, trainingSize + 1):
            feature = SignatureFeatureExtraction.preprocess_and_feature_extraction_radon_transform_features(f'{location}/signature{i}.png')
            featureList.append(feature)  # append each feature to the list
        config.global_features = featureList    
        utils = Utilities()
        dtw_distance = utils.compute_training_score(featureList)
        #print("Computed DTW Distance:", dtw_distance)
        return dtw_distance
    
    




