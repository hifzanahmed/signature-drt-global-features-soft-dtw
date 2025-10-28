import numpy as np
import cv2
from tslearn.metrics import soft_dtw

class Utilities:
    @staticmethod
    def load_image(image_path):
        import cv2
        # Load the image
        signature = cv2.imread(image_path)
        
        # Check if image is loaded correctly
        if signature is None:
            print("Error: Signature Image not found or unable to load.")
            return None
        else:
            #print("Signature Image loaded successfully.")
            # Show the image
            cv2.imshow("Signature Image", signature)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return signature
        
    @staticmethod
    def crop_signature(binary_img):
        """
        Crops the binary image by removing all-zero rows and columns.
        Args:
        binary_img (np.ndarray): 2D numpy array where 1s represent signature, 0s are background.
        Returns:
        np.ndarray: Cropped binary image.
        """
        assert binary_img.ndim == 2, "Input must be a 2D array"
        # Find rows and columns with at least one '1'
        rows = np.any(binary_img, axis=1)
        cols = np.any(binary_img, axis=0)
        # Crop the image
        cropped_img = binary_img[rows][:, cols]
        return cropped_img
    
    @staticmethod
    def crop_and_resize_signature(binary_img):
        """
        Crops the binary image by removing all-zero rows and columns.
        Args:
        binary_img (np.ndarray): 2D numpy array where 1s represent signature, 0s are background.
        Returns:
        np.ndarray: Cropped binary image.
        """
        assert binary_img.ndim == 2, "Input must be a 2D array"
        # Find rows and columns with at least one '1'
        rows = np.any(binary_img, axis=1)
        cols = np.any(binary_img, axis=0)
        # Crop the image
        cropped_img = binary_img[rows][:, cols]
        return Utilities.resize_image(cropped_img)
    
    @staticmethod
    def resize_image(img, size=(300, 150)):
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def extract_features(cropped_img):
        processed_image_features = Utilities.horizontal_vertical_projection(cropped_img)
        #print("Horizontal and Vertical Projection:", processed_image_features.size)
        return processed_image_features
    
    @staticmethod
    def extract_features_discrete_radon_transform(cropped_img):
        processed_image_features = Utilities.horizontal_vertical_projection_discrete_radon_transform(cropped_img)
        print("Horizontal and Vertical Projection:", processed_image_features.size)
        return processed_image_features

    @staticmethod
    def horizontal_projection(binary_img):
        """
        Calculate horizontal projection of a binary image.
        Args:
        binary_img (np.ndarray): 2D binary image (0s and 1s)
        Returns:
        np.ndarray: 1D array containing sum of pixels in each row
        """
        assert binary_img.ndim == 2, "Image must be 2D"
        return np.sum(binary_img, axis=1)
    
    @staticmethod
    def vertical_projection(binary_img):
        """
        Calculate vertical projection of a binary image.
        Args:
        binary_img (np.ndarray): 2D binary image (0s and 1s)
        Returns:
        np.ndarray: 1D array containing sum of pixels in each row
        """
        assert binary_img.ndim == 2, "Image must be 2D"
        return np.sum(binary_img, axis=0)
    
    @staticmethod
    def horizontal_vertical_projection(binary_img):
        """
        Calculate horizontal and vertical projections of a binary image.
        Args:
        binary_img (np.ndarray): 2D binary image (0s and 1s)
        Returns:
        tuple: (horizontal_proj, vertical_proj)
            - horizontal_proj: 1D array of sums across rows
            - vertical_proj: 1D array of sums across columns
        """
        assert binary_img.ndim == 2, "Input must be 2D array"
        horizontal_proj = np.sum(binary_img, axis=1)
        vertical_proj = np.sum(binary_img, axis=0)
        return np.concatenate((horizontal_proj, vertical_proj))
    
    @staticmethod
    def discrete_radon_transform(binary_img):
        # 0° projection (horizontal features): sum along rows
        horizontal_projection = np.sum(binary_img, axis=1)
        # 90° projection (vertical features): sum along columns
        vertical_projection = np.sum(binary_img, axis=0)
        return horizontal_projection, vertical_projection
    
    @staticmethod
    def horizontal_vertical_projection_discrete_radon_transform(binary_img):
        horizontal_proj, vertical_proj = Utilities.discrete_radon_transform(binary_img)
        return np.concatenate((horizontal_proj, vertical_proj))
    
    @staticmethod
    def compute_training_score(signatures):
        """
        Computes S1: average Soft-DTW distance between all pairs of genuine signatures.
        Args:
            signatures (list of np.ndarray): List of K signature samples (1D arrays)
        Returns:
            float: Average Soft-DTW distance (S1)
        """
        K = len(signatures)
        dist_matrix = np.zeros((K, K))
        utils = Utilities()  
        print("signature list size:", K)
        for i in range(K):
            for j in range(i + 1, K):
                d = utils.soft_dtw(
                    signatures[i].reshape(-1, 1),
                    signatures[j].reshape(-1, 1),
                    gamma=1.0
                )
                dist_matrix[i, j] = dist_matrix[j, i] = d

        print("Pairwise Soft-DTW distance matrix:\n", dist_matrix)
        # Average distance between all unique pairs (i < j)
        avg_distance = np.sum(np.triu(dist_matrix, k=1)) / (K * (K - 1) / 2)
        return avg_distance
    
    @staticmethod
    def compute_verification_score(test_signature, genuine_signatures):
        """
        Computes S2: average Soft-DTW distance between a test signature and all genuine signatures.
        Args:
            test_signature (np.ndarray): 1D array of the test signature features.
            genuine_signatures (list of np.ndarray): List of genuine signature features.
            gamma (float): Soft-DTW smoothing parameter.
        Returns:
            float: Average Soft-DTW distance (S2 score).
        """
        K = len(genuine_signatures)
        if K == 0:
            return 0.0  # Avoid division by zero

        utils = Utilities()
        total_dist = 0.0
        gamma = 1.0

        for i in range(K):
            dist = utils.soft_dtw(
                test_signature.reshape(-1, 1),
                genuine_signatures[i].reshape(-1, 1),
                gamma=gamma
            )
            total_dist += dist

        avg_dist = total_dist / K
        return avg_dist

    @staticmethod
    def soft_dtw(signature1, signature2, gamma=1.0):
        print("Training on Genuine Signatures...")
        # Soft-DTW gamma parameter (controls smoothness)
        # Compute soft-DTW distance
        soft_dtw_distance = soft_dtw(signature1.reshape(-1, 1), signature2.reshape(-1, 1), gamma=gamma)
        print("Soft-DTW distance:", soft_dtw_distance)
        return soft_dtw_distance