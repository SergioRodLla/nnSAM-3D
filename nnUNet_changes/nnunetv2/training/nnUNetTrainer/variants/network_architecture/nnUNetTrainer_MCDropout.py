from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn
from nnunetv2.utilities.get_network_from_plans_plusDropout import get_network_from_plans_dropout

class nnUNetTrainer_MCDropout(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   enable_deep_supervision: bool = True,) -> nn.Module:
        model = get_network_from_plans_dropout(
            plans_manager = plans_manager,
            dataset_json = dataset_json,
            configuration_manager = configuration_manager,
            num_input_channels = num_input_channels,
            num_output_channels = num_output_channels,
            deep_supervision = enable_deep_supervision)
        
        print("Using model for MCDropout", "\n")
        #print(model) # ensure new model archictecture is correctly set!
        return model

