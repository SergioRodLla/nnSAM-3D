import os
import sys
import numpy as np
import torch
from torch._dynamo import OptimizedModule

import SimpleITK as sitk
#from batchgenerators.utilities.file_and_folder_operations import join
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from typing import Tuple, Union, List, Optional

import nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


# Enable Dropout during inference (model.eval())
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def predict_with_uncertainty(predictor: nnUNetPredictor, 
                             input_folder: str, 
                             output_folder: str, 
                             num_fwd_passes: int,
                             simulate_models: bool = True,
                             compute_entropy: bool = False) -> None:
    """
    - Set model.train() to keep dropout active during each forward pass.
    - Perform multiple forward passes for each test image, simulating different models.
    - Save each prediction in separate directories to simulate different models.
    - Calculate Entropy for each patient
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a folder for storing entropy results
    entropy_folder = os.path.join(output_folder, 'entropy')
    if not os.path.exists(entropy_folder):
        os.makedirs(entropy_folder)

    model_dirs = []
    for i in range(num_fwd_passes):
        #i += 15 # !!!!!!!quick fix to generate more folders!!!!!!!!!
        # model_dir = f"model_{i}"
        #model_dir = f"EnableDropout_{i}"
        model_dir = f"preds_{i}"
        model_path = os.path.join(output_folder, model_dir)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_dirs.append(model_path)
    
    patient_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
    patient_ids = list(set([os.path.basename(f).rsplit('_', 1)[0] for f in patient_files])) # get patients' unique IDs

    if simulate_models:
        # files must be given as 'list of lists' where each entry in the outer list is a case to be predicted and the inner list contains all the files belonging to that case
        for pat_id in patient_ids:
            #if pat_id in ['CHUP_032', 'CHUP_070', 'CHUS_022']:
            patient_files_for_id = [file for file in patient_files if pat_id in file]
            # sort the list so that the file ending with _0000.nii.gz is always first
            patient_files_for_id.sort(key=lambda x: x.endswith('_0000.nii.gz'), reverse=True)
            print(patient_files_for_id)
            for i, model_dir in enumerate(model_dirs):
                predictor.network.eval()
                enable_dropout(predictor.network) # Enable dropout
                with torch.no_grad():
                    pred = predictor.predict_from_files(
                        list_of_lists_or_source_folder=[patient_files_for_id], # has to be 'list of lists'
                        output_folder_or_list_of_truncated_output_files=None,
                        save_probabilities=True,
                        overwrite=True,
                        num_processes_preprocessing=1,
                        num_processes_segmentation_export=1,
                        folder_with_segs_from_prev_stage=None,
                        num_parts=1,
                        part_id=0
                    )
                    #pred[0][1].shape = (3, 176, 176, 176)

                    pred_class_probs = pred[0][1]

                    # Save logits as npz
                    np.savez_compressed(os.path.join(model_dir, pat_id + '_logits.npz'), pred_class_probs)
                    
                    # Save prediction as NIfTI
                    pred_class = np.argmax(pred_class_probs, axis=0)
                    pred_image = sitk.GetImageFromArray(pred_class.astype(np.uint8))
                    reference_image = sitk.ReadImage(patient_files_for_id[0]) # use CT modality as reference image
                    pred_image.CopyInformation(reference_image)
                    sitk.WriteImage(pred_image, os.path.join(model_dir, pat_id+'.nii.gz'))
            print(f"Preds and Logits done for {pat_id}")

    # Calculate and save entropy for each patient
    if compute_entropy:
        for pat_id in patient_ids:
            predictions = []
            for model_dir in model_dirs:
                pred = np.load(os.path.join(model_dir, pat_id+'_logits.npz'))['arr_0']
                predictions.append(pred)
            
            predictions = np.stack(predictions)
            mean_prediction = predictions.mean(axis=0)
            entropy = -np.sum(mean_prediction * np.log(mean_prediction + 1e-6), axis=0)

            # Save entropy
            entropy_image = sitk.GetImageFromArray(entropy)
            reference_image = sitk.ReadImage(os.path.join(input_folder, f'{pat_id}_0000.nii.gz'))  # use CT modality as reference
            entropy_image.CopyInformation(reference_image)
            sitk.WriteImage(entropy_image, os.path.join(entropy_folder, f'{pat_id}_entropy.nii.gz'))
            print(f"Entropy done for {pat_id}")

def verify_dropout(predictor: nnUNetPredictor) -> None:
    # set network to evaluation mode
    predictor.network.eval()
    # activate only dropout layers
    enable_dropout(predictor.network)

    # verify dropout layers' status
    for m in predictor.network.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print(f"Layer {m.__class__.__name__} is in training mode: {m.training}")  # should be True
            # if not m.training:
            #     raise SystemExit("Dropout layers are not activated (model.train())!!! \n STOPPING...")
        else:
            print(f"Layer {m.__class__.__name__} is in training mode: {m.training}")  # should be False


# Change nnUNerPredictor for STUNet implementation
class STUNet_predictor(nnUNetPredictor):
    def __init__(self, tile_step_size: float = 0.5, use_gaussian: bool = True, use_mirroring: bool = True, perform_everything_on_device: bool = True, device: torch.device = ..., verbose: bool = False, verbose_preprocessing: bool = False, allow_tqdm: bool = True):
        super().__init__(tile_step_size, use_gaussian, use_mirroring, perform_everything_on_device, device, verbose, verbose_preprocessing, allow_tqdm)
    
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        
        # !! CHANGE HERE: build_network_architecture needs to be called according to STUNetTrainer arguments
        network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                           num_input_channels, enable_deep_supervision=False)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)


def main() -> None:
    model_folder = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/data/nnUNet_results/Dataset500_HeadNeckPTCT/STUNetTrainer_base_ft__nnUNetPlans__3d_fullres"
    input_dir = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/data/nnUNet_raw_data/Dataset500_HeadNeckPTCT/imagesTs"
    output_dir = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/holdout_STUNet_base_preds"

    # Initialize the nnUNetPredictor
    # Create new custom nnUNetPredictor that works with STUNet implementation !!
    predictor = STUNet_predictor(
        device=torch.device('cuda', 0)
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth'
    )

    # Perform inference with uncertainty estimation
    #verify_dropout(predictor) # works as expected
    predict_with_uncertainty(predictor, input_dir, output_dir, num_fwd_passes=1, simulate_models=True, compute_entropy=False)

if __name__ == "__main__":
    main()
