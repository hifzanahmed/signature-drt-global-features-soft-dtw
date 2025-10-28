from utilities import Utilities
from signature_feature_extraction import SignatureFeatureExtraction
import config
import numpy as np
import torch

class SignatureVerificationTraining:
    def verifiy_test_signature_with_soft_dtw_without_gradient(test_image_path):
        test_feature = SignatureFeatureExtraction.preprocess_and_feature_extraction_radon_transform_features(test_image_path)
        utils = Utilities()
        dtw_distance = utils.compute_verification_score(test_feature, config.global_features)
        #print("Computed DTW Distance:", dtw_distance)
        return dtw_distance