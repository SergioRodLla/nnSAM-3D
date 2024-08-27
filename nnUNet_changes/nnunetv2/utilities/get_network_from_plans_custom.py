# From uncertainty-segmentation-mcdropout --> segmentation --> modification_nnunet --> get_network_from_plans.py
# This module name (filename) has been changed to keep the original one

# From original code: https://github.com/MIC-DKFZ/nnUNet
# nnunetv2 --> utilities --> get_network_from_plans.py (line 40)

from nnunetv2.utilities.unet import SAMConvUNet, SAM3DConvUNet
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
import os

def get_network_from_plans_custom(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           num_output_channels: int, 
                           drop_prob: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    # num_stages = len(configuration_manager.conv_kernel_sizes)

    # dim = len(configuration_manager.conv_kernel_sizes[0])

    arch_kwargs = configuration_manager.network_arch_init_kwargs
    num_stages = len(arch_kwargs['kernel_sizes'])

    dim = len(arch_kwargs['kernel_sizes'][0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    #segmentation_network_class_name = configuration_manager.UNet_class_name
    segmentation_network_class_name = configuration_manager.network_arch_class_name

    # Get nnSAM architecture
    if os.environ.get('MODEL_NAME') == 'nnsam_2d':
        segmentation_network_class_name = 'SAMConvUNet'
    #assert os.environ.get('MODEL_NAME') == 'nnsam_2d', "The trainer specified is nnSAM_Trainer but MODEL_NAME was not set to nnsam_2d"
        
    
    if os.environ.get('MODEL_NAME') == 'nnsam_3d':
        segmentation_network_class_name = 'SAM3DConvUNet'
    #assert os.environ.get('MODEL_NAME') == 'nnsam_3d', "The trainer specified is nnSAM3D_Trainer but MODEL_NAME was not set to nnsam_3d"
        

    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet,
        'dynamic_network_architectures.architectures.unet.PlainConvUNet': PlainConvUNet,
        'dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet': ResidualEncoderUNet,
        'SAMConvUNet': SAMConvUNet,
        'SAM3DConvUNet': SAM3DConvUNet # set nnSAM3D with our custom architecture
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': nn.Dropout3d, 'dropout_op_kwargs': {'p':drop_prob}, #change here!
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'dynamic_network_architectures.architectures.unet.PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': nn.Dropout3d, 'dropout_op_kwargs': {'p': drop_prob},  # change here!
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'SAMConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'SAM3DConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }    
    }
    #print(f"Error here: {segmentation_network_class_name}")
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage' if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': arch_kwargs['n_conv_per_stage'],
        'n_conv_per_stage_decoder': arch_kwargs['n_conv_per_stage_decoder']
    }
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=arch_kwargs['features_per_stage'],
        conv_op=conv_op,
        kernel_sizes=arch_kwargs['kernel_sizes'],
        strides=arch_kwargs['strides'],
        num_classes=label_manager.num_segmentation_heads,
        #num_classes=num_output_channels,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model