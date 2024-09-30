import os
import torch
import itertools
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from typing import Tuple, Union, List, Optional

import nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

# Import SAM-Med3D model
import sys
sys.path.append('/media/HDD_4TB_2/sergio/TFM/SAM-Med3D/SAM-Med3D')
from segment_anything.build_sam3D import sam_model_registry3D

class SAMMed3D_Predictor(nnUNetPredictor):
    def __init__(self, tile_step_size: float = 0.5, use_gaussian: bool = True, use_mirroring: bool = True, perform_everything_on_device: bool = True, device: torch.device = ..., verbose: bool = False, verbose_preprocessing: bool = False, allow_tqdm: bool = True):
        super().__init__(tile_step_size, use_gaussian, use_mirroring, perform_everything_on_device, device, verbose, verbose_preprocessing, allow_tqdm)
        
    def initialize_from_trained_model_folder(self, *args, **kwargs):
        """
        This is used when making predictions with a trained model.
        We override this to load the SAM-Med3D model instead of the original nnUNet model.
        """

        checkpoint_path = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/data/nnUNet_raw_data/sam_med3d_turbo.pth"

        # Plans file from hecktor with only CT
        plans_file = '/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/data/nnUNet_preprocessed/Dataset501_HecktorCT/nnUNetPlans.json'
        
        # Dataset file from hecktor with only CT
        dataset_file = '/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/data/nnUNet_preprocessed/Dataset501_HecktorCT/dataset.json'

        # Load plans and dataset json
        dataset_json = load_json(dataset_file)
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)

        configuration_name = "3d_fullres"
        configuration_manager = plans_manager.get_configuration(configuration_name)

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        num_classes=plans_manager.get_label_manager(dataset_json).num_segmentation_heads

        # Define SAM-Med3D model
        # IMPORTANT: Here I am skipping the usual build_network_architecture() method to define the network since there is no trainer for SAM-Med3D
        model_type = "vit_b_ori"
        network = sam_model_registry3D[model_type](checkpoint=None)
        
        # Load SAM-Med3D checkpoint weights
        parameters = []
        sammed3d_checkpoint = torch.load(checkpoint_path, map_location=self.device)
        parameters.append(sammed3d_checkpoint['model_state_dict']) # Check this

        # Store the network and its parameters
        self.network = network
        self.list_of_parameters = parameters
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.trainer_name = 'SAMMed3D_Trainer'  # Just a placeholder name it doesn't really exist
        self.allowed_mirroring_axes = None  # Adjust this if needed
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, torch._dynamo.OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)
    
    # Atempt to fix:
    # Modify this from original to pass 'multimask_output' argument to Sam3D.forward()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input has only one channel and expand it to 3 channels
        if x.shape[1] == 1:
            print("Expanding channels from 1 to 3")
            x = x.repeat(1, 3, 1, 1, 1)  # Shape becomes (batch, 3, depth, height, width)
        
        # Modify the pixel_mean and pixel_std to match the expanded tensor assuming the mean and std are originally meant for 3 channels
        original_pixel_mean = self.network.pixel_mean
        original_pixel_std = self.network.pixel_std
    
        mean = torch.tensor(original_pixel_mean).view(1, 3, 1, 1, 1)
        std = torch.tensor(original_pixel_std).view(1, 3, 1, 1, 1)
        
        # Normalize the tensor manually
        x = (x - mean) / std
        
        # Print the shape for debugging
        print(f"Input shape after channel expansion and normalization: {x.shape}")

        # Prepare the input as expected by SAM3D
        batched_input = [{"image": x}]

        try:
            prediction = self.network(batched_input, multimask_output=False)
        except RuntimeError as e:
            print(f"Prediction on device was unsuccessful, due to: {str(e)}")
            raise

        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None

        if mirror_axes is not None:
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                flipped_input = [{"image": torch.flip(x, axes)}]
                prediction += torch.flip(self.network(flipped_input, multimask_output=False), axes)
            prediction /= (len(axes_combinations) + 1)

        return prediction






if __name__ == "__main__":
    from nnunetv2.paths import nnUNet_results, nnUNet_raw

    test_img_path = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/data/nnUNet_raw_data/Dataset501_HecktorCT/imagesTs"
    output_dir_path = "/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/SAM-Med3D_inference"

    predictor = SAMMed3D_Predictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder()
    print("SAM-Med3D predictor initialized correctly")

    predictor.predict_from_files(test_img_path,
                                 output_dir_path,
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

