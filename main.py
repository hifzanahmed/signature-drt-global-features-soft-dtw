from training.signature_training import SignatureTraining
from verification.signature_verification import SignatureVerificationTraining
import os

def main():
    location_of_training_signature = 'C:/Users/hifza/workspace/Signature Dataset/za/'
    size_of_training_signature = 6
    location_of_test_signature = 'C:/Users/hifza/workspace/Signature Dataset/za/forged'
    # Training Phase
    s1 = SignatureTraining.training_genuine_with_soft_dtw_without_gradient(location_of_training_signature, size_of_training_signature)  
    # Verification Phase of input test signature 
    # Loop through sequentially named images: image1, image2, ...
    i = 1
    while True:
        test_signature_path = os.path.join(location_of_test_signature, f"signature{i}.png")
    
        # Stop if the image does not exist
        if not os.path.exists(test_signature_path):
            break
        s2 = SignatureVerificationTraining.verifiy_test_signature_with_soft_dtw_without_gradient(test_signature_path)
        # Decision Making: calculating the score and comparing it with a threshold value
        i += 1
        score_ratio = abs(s2) / abs(s1)
        print("test_signature_path:", test_signature_path) 
        print("S1 (Training Score):", s1) 
        print("S2 (Verification Score):", s2)
        print("score_ratio for verification:", score_ratio)
        if score_ratio > 1:  
            print("Genuine Signature")
        else:
            print("Forged Signature")

if __name__ == "__main__":
    main()