'''
This script contains the trainer classes to use the following network architectures:
    - nnUNetTrainer_MCD: adds dropout layers to the baseline nnUNet model (specify drop probabilty)
    - nnSAM_Trainer: original nnSAM model that can process 2D inputs
    - nnSAM3D_Trainer: our custom model leveraging SAM-Med3D image encoder

To use them call nnUNetv2_train with the trainer class name with "-tr", for example:
    nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD -tr nnSAM3D_Trainer

All of these inherit from the nnUNetTrainer class.
They build the desired architecture by calling get_network_from_plans_custom.

IMPORTANT: To use nnSAM-3D model require setting enviromental variable like:
    export MODEL_NAME=nnsam_3d
In this way, the function gets this information

'''


from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from torch import nn
from nnunetv2.utilities.get_network_from_plans_custom import get_network_from_plans_custom
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

# trainer class to build the MCD architecture specifying drop prob.
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
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model = get_network_from_plans_custom(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=configuration_manager,
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            drop_prob=0.2,
            deep_supervision=enable_deep_supervision
        )
        
        print("Using model for MCDropout with drop prob. 0.2", "\n")
        print(model) # ensure new model archictecture is correctly set!
        return model


    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
        
    # def initialize(self):
    #     # Ensure that dataset_json and configuration_manager are accessible in this method
    #     self.network = self.build_network_architecture(
    #         architecture_class_name=self.configuration_manager.network_arch_class_name,
    #         arch_init_kwargs=self.configuration_manager.network_arch_init_kwargs,
    #         arch_init_kwargs_req_import=self.configuration_manager.network_arch_init_kwargs_req_import,
    #         num_input_channels=self.num_input_channels,
    #         num_output_channels=self.label_manager.num_segmentation_heads,
    #         plans_manager=self.plans_manager,
    #         dataset_json=self.dataset_json,
    #         configuration_manager=self.configuration_manager,
    #         enable_deep_supervision=self.enable_deep_supervision
    #     )

# trainer class to build the nnSAM (2D) architecture
class nnSAM_Trainer(nnUNetTrainer):
    '''
    IMPORTANT: Using nnSAM architecture requites setting enviromental variable like this:
    export MODEL_NAME=nnsam_2d
    '''
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model = get_network_from_plans_custom(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=configuration_manager,
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            drop_prob=0.5, # there are no dropout layers but keeping this for now
            deep_supervision=enable_deep_supervision
        )
        
        print("Using nnSAM (2D) architecture", "\n")
        print(model) # ensure new model archictecture is correctly set!
        return model


    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
        
# trainer class to build the nnSAM (2D) architecture
class nnSAM3D_Trainer(nnUNetTrainer):
    '''
    IMPORTANT: Using nnSAM3D architecture require setting enviromental variable like this:
    export MODEL_NAME=nnsam_3d
    '''
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model = get_network_from_plans_custom(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=configuration_manager,
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            drop_prob=0.5, # there are no dropout layers but keeping this for now
            deep_supervision=enable_deep_supervision
        )
        
        print("Using nnSAM3D architecture", "\n")
        print(model) # ensure new model archictecture is correctly set!
        return model


    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")