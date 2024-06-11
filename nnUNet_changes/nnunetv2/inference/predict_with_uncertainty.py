import os
import numpy as np
import torch
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def predict_with_uncertainty(predictor: nnUNetPredictor, 
                             input_folder: str, 
                             output_folder: str, 
                             num_fwd_passes: int = 15) -> None:
    """
    - Set model.train() to keep dropout active during each forward pass.
    - Perform multiple forward passes for each test image, simulating different models.
    - Save each prediction in separate directories to simulate different models.
    - Calculate Entropy for each patient
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_dirs = []
    for i in range(num_fwd_passes):
        model_dir = f"model_{i}"
        model_path = os.path.join(output_folder, model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_dirs.append(model_path)
    
    patient_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.nii.gz')]

    for patient_file in patient_files:
        for i, model_dir in enumerate(model_dirs):
            predictor.network.train()  # Enable dropout
            with torch.no_grad():
                pred = predictor.predict_from_files(
                    list_of_lists_or_source_folder=[[patient_file]],
                    output_folder_or_list_of_truncated_output_files=None,
                    save_probabilities=True,
                    overwrite=True,
                    num_processes_preprocessing=1,
                    num_processes_segmentation_export=1,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0
                )[0]  # Assuming predict_from_files returns a list and we need the first item
                
                # Save logits as npz
                np.savez_compressed(os.path.join(model_dir, os.path.basename(patient_file).replace('.nii.gz', '_logits.npz')), pred)
                
                # Save prediction as NIfTI
                pred_class = np.argmax(pred, axis=0)
                pred_image = sitk.GetImageFromArray(pred_class.astype(np.uint8))
                reference_image = sitk.ReadImage(patient_file)
                pred_image.CopyInformation(reference_image)
                sitk.WriteImage(pred_image, os.path.join(model_dir, os.path.basename(patient_file)))
        print(f"Preds and Logits done for {os.path.basename(patient_file).replace('.nii.gz', '')}")

    # Calculate and save entropy for each patient
    for patient_file in patient_files:
        predictions = []
        for model_dir in model_dirs:
            pred = np.load(os.path.join(model_dir, os.path.basename(patient_file).replace('.nii.gz', '_logits.npz')))['arr_0']
            predictions.append(pred)
        
        predictions = np.stack(predictions)
        mean_prediction = predictions.mean(axis=0)
        entropy = -np.sum(mean_prediction * np.log(mean_prediction + 1e-6), axis=0)

        # Save entropy
        entropy_image = sitk.GetImageFromArray(entropy)
        reference_image = sitk.ReadImage(patient_file)
        entropy_image.CopyInformation(reference_image)
        sitk.WriteImage(entropy_image, os.path.join(output_folder, os.path.basename(patient_file).replace('.nii.gz', '_entropy.nii.gz')))
        print(f"Entropy done for {os.path.basename(patient_file).replace('.nii.gz', '')}")

def main() -> None:
    model_folder = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/data/nnUNet_results/Dataset500_HeadNeckPTCT/nnUNetTrainer_MCDropout__nnUNetPlans__3d_fullres"
    input_dir = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/data/nnUNet_raw_data/Dataset500_HeadNeckPTCT/imagesTs"
    output_dir = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/test_preds_MCD"

    # Initialize the nnUNetPredictor
    predictor = nnUNetPredictor(
        device=torch.device('cuda', 0)
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth'
    )

    # Perform inference with uncertainty estimation
    predict_with_uncertainty(predictor, input_dir, output_dir, num_fwd_passes=15)

if __name__ == "__main__":
    main()
