from training.signature_training import SignatureTraining
from verification.signature_verification import SignatureVerificationTraining
import os

def main():
    location_of_training_signature = 'C:/Users/hifza/workspace/Signature Dataset/Saba/signature'
    size_of_training_signature = 6
    location_of_test_signature = 'C:/Users/hifza/workspace/Signature Dataset/Saba/Forged/signature'
    # Training Phase
    s1 = SignatureTraining.training_genuine_with_soft_dtw_without_gradient(location_of_training_signature, size_of_training_signature)  
    # Verification Phase of input test signature and Loop through sequentially named images: image1, image2, ...
    i = 1
    while True:
        test_signature = f"{location_of_test_signature}{i}.png"
        test_signature_path = os.path.join(test_signature)
    
        # Stop if the image does not exist
        if not os.path.exists(test_signature_path):
            break
        s2 = SignatureVerificationTraining.verifiy_test_signature_with_soft_dtw_without_gradient(test_signature_path)
        # Decision Making: calculating the score and comparing it with a threshold value
        i += 1
        score = abs(s1) / abs(s2)
        if score < 1:  
            print(f"{score:.4f};Genuine")
        else:
            print(f"{score:.4f};Forged")

if __name__ == "__main__":
    main()