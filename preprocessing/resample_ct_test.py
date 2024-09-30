
from pathlib import Path
from multiprocessing import Pool
import logging

import click
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os 
import sys
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk


def main_resample(arguments):
    """ This command line interface allows to resample NIFTI files within a
        given bounding box contain in BOUNDING_BOXES_FILE. The images are
        resampled with spline interpolation
        of degree 3 and the segmentation are resampled
        by nearest neighbor interpolation.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """
    p, input_folder,input_label_folder, output_folder, bb_df = arguments
    resampling=(1, 1, 1)
    output_folder = Path(output_folder)
    input_folder = Path(input_folder)
    input_label_folder = Path(input_label_folder)
    output_folder.mkdir(exist_ok=True)

    print(f'Patient {p} started to resample...')
    sys.stdout.flush()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    bb = np.array([
        bb_df.loc[p, 'x1'] - 24, bb_df.loc[p, 'y1'] - 12, bb_df.loc[p, 'z1'] - 48,
        bb_df.loc[p, 'x2'] + 24, bb_df.loc[p, 'y2'] + 36, bb_df.loc[p, 'z2']
    ])
    size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
    ct = sitk.ReadImage(
        str([f for f in input_folder.rglob(p + "__CT*")][0].resolve()))
    # pt = sitk.ReadImage(
    #     str([f for f in input_folder.rglob(p + "__PT*")][0].resolve()))
    gtvt = sitk.ReadImage(
        str([f for f in input_label_folder.rglob(p + "*")][0].resolve()))
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])  # sitk is so stupid
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct = resampler.Execute(ct)
    # pt = resampler.Execute(pt)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    gtvt = resampler.Execute(gtvt)

    print(f'Patient {p} resample completed. ')
    sys.stdout.flush()

    sitk.WriteImage(ct, str(
        (output_folder / (p + "__CT.nii.gz")).resolve()))
    # sitk.WriteImage(pt, str(
    #     (output_folder / (p + "__PT.nii.gz")).resolve()))
    sitk.WriteImage(gtvt,
                    str((output_folder / (p + "__gtv.nii.gz")).resolve()))
    print(f'Patient {p} saved.')
    sys.stdout.flush()

# def main_revert(arguments):
#     bb, segmentation_path, original_image_path, output_path = arguments

#     # Load the original image
#     original_image = sitk.ReadImage(str(original_image_path))

#     # Load the predicted mask image
#     predicted_mask = sitk.ReadImage(str(segmentation_path))

#     # Create an intermediate image that is blank everywhere except within the bounding box
#     intermediate_image = sitk.Image(original_image.GetSize(), sitk.sitkUInt8)
#     intermediate_image.SetSpacing(original_image.GetSpacing())
#     intermediate_image.SetOrigin(original_image.GetOrigin())
#     #intermediate_image_array = sitk.GetArrayFromImage(intermediate_image)
#     #print(np.max(intermediate_image_array), intermediate_image_array.shape)
    
#     print(original_image.GetSpacing(), predicted_mask.GetSpacing() )

#     # Make size for the intermediate image to full size
#     full_size = [int(original_image.GetSize()[i] * original_image.GetSpacing()[i]) for i in range(3)] # to make image to 1x1x1
#     print(full_size, original_image.GetSize())

#     # Resample the intermediate image to match the spaceing of the predicted_mask
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetOutputDirection(predicted_mask.GetDirection())
#     resampler.SetOutputSpacing(predicted_mask.GetSpacing())
#     resampler.SetSize(full_size)
#     resampler.SetOutputOrigin(original_image.GetOrigin())
#     resampler.SetInterpolator(sitk.sitkNearestNeighbor)
#     resampled_intermediate_image = resampler.Execute(intermediate_image)

#     # Calculate the bounding box size and the offset 
#     size = [int((bb[i + 3] - bb[i])) for i in range(3)] 
#     offset = [int(bb[i] - original_image.GetOrigin()[i]) for i in range(3)]

#     predicted_mask_array = sitk.GetArrayFromImage(predicted_mask)
#     print(np.max(predicted_mask_array), predicted_mask_array.shape, size)

#     # Paste the resampled predicted mask into the intermediate image
#     intermediate_image = sitk.Paste(resampled_intermediate_image, predicted_mask, size, [0,0,0], offset ) #[0,0,0], offset)

#     # Resample the intermediate image to the original image spacing and size
#     resampler.SetOutputSpacing(original_image.GetSpacing())
#     resampler.SetOutputDirection(original_image.GetDirection())
#     resampler.SetSize(original_image.GetSize())
#     resampler.SetOutputOrigin(original_image.GetOrigin())
#     resampler.SetInterpolator(sitk.sitkNearestNeighbor)
#     resampled_mask = resampler.Execute(intermediate_image)

#     resampled_mask_array = sitk.GetArrayFromImage(resampled_mask)
#     print(np.max(resampled_mask_array), resampled_mask_array.shape)
#     # Write the reverted segmentation to disk
#     sitk.WriteImage(resampled_mask, str(output_path))

def main_revert(arguments):
    """
    Reverts a predicted mask to match the original image size and spacing.
    
    Arguments:
    - arguments: A list of arguments containing the bounding box coordinates, segmentation path,
                 original image path, and output path.

    Returns:
    None
    """
    bb, segmentation_path, original_image_path, output_path = arguments

    # Load the original image
    original_image = sitk.ReadImage(str(original_image_path))

    # Load the predicted mask image
    predicted_mask = sitk.ReadImage(str(segmentation_path))

    # Create an intermediate image that is blank everywhere except within the bounding box
    intermediate_image = sitk.Image(original_image.GetSize(), sitk.sitkUInt8)
    intermediate_image.SetSpacing(original_image.GetSpacing())
    intermediate_image.SetOrigin(original_image.GetOrigin())

    print(original_image.GetSpacing(), predicted_mask.GetSpacing() )

    patch_size = [int((bb[i + 3] - bb[i])/original_image.GetSpacing()[i]) for i in range(3)] 
    # Resample the intermediate image to match the spaceing of the predicted_mask
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(original_image.GetDirection())
    resampler.SetOutputSpacing(original_image.GetSpacing())
    resampler.SetSize(patch_size)
    resampler.SetOutputOrigin(predicted_mask.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_intermediate_image = resampler.Execute(predicted_mask)

    # Calculate the bounding box size and the offset 
    size = [int((bb[i + 3] - bb[i])) for i in range(3)] 
    offset = [int(bb[i] - original_image.GetOrigin()[i]) for i in range(3)]

    predicted_mask_array = sitk.GetArrayFromImage(predicted_mask)
    print(np.max(predicted_mask_array), predicted_mask_array.shape, size)

    # Paste the resampled predicted mask into the intermediate image
    intermediate_image = sitk.Paste(resampled_intermediate_image, predicted_mask, size, [0,0,0], offset ) #[0,0,0], offset)

    # Resample the intermediate image to the original image spacing and size
    resampler.SetOutputSpacing(original_image.GetSpacing())
    resampler.SetOutputDirection(original_image.GetDirection())
    resampler.SetSize(original_image.GetSize())
    resampler.SetOutputOrigin(original_image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = resampler.Execute(intermediate_image)

    resampled_mask_array = sitk.GetArrayFromImage(resampled_mask)
    print(np.max(resampled_mask_array), resampled_mask_array.shape)
    # Write the reverted segmentation to disk
    sitk.WriteImage(resampled_mask, str(output_path))


def resample_images(input_folder, input_label_folder, output_folder, bounding_boxes_file):
    # Load bounding box information
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    
    for p in bb_df.index:
        # Resample images
        main_resample((p, input_folder, input_label_folder, output_folder, bb_df))

def revert_segmentation(segmentation_path, bounding_boxes_file, original_image_path, output_path, pat_list):
    # Load bounding box information
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')

    #for patient in bb_df.index:
    for patient in pat_list:
        bb = np.array([
            bb_df.loc[patient, 'x1'] - 24, bb_df.loc[patient, 'y1'] - 12, bb_df.loc[patient, 'z1'] - 48,
            bb_df.loc[patient, 'x2'] + 24, bb_df.loc[patient, 'y2'] + 36, bb_df.loc[patient, 'z2']
        ])

        segmentation_file = os.path.join(segmentation_path, patient+'.nii.gz')
        print(original_image_path)
        image_path = os.path.join(original_image_path, patient+'__CT.nii.gz')
        print(image_path)
        output_file = os.path.join(output_path, patient+'.nii.gz')
        # Revert the segmentation
        main_revert((bb, segmentation_file, image_path, output_file))


    
if __name__ == "__main__":
    """
    This script uses argparse to parse command line arguments. The --mode argument specifies which operation to perform:
      'resample' or 'revert'. Depending on the mode, it will either call resample_images or revert_segmentation.

    In 'resample' mode, it will call resample_images, which in turn calls main_resample for each patient in the bounding 
    boxes file.

    In 'revert' mode, it will call revert_segmentation, which in turn calls main_revert to revert the segmentation for a 
    single patient.
    """

    parser = argparse.ArgumentParser(description='Resample and revert segmentations.')
    parser.add_argument('--mode', type=str, choices=['resample', 'revert'], help='Mode to run: "resample" or "revert".')
    parser.add_argument('--input_folder', type=str, help='Input folder containing images.')
    parser.add_argument('--input_label_folder', type=str, help='Input folder containing labels.')
    parser.add_argument('--output_folder_resample', type=str, help='Output folder to save resampled images/labels.')
    parser.add_argument('--bounding_boxes_file', type=str, help='CSV file containing bounding boxes.')
    parser.add_argument('--segmentation_path', type=str, help='Path to the predicted segmentation.')
    parser.add_argument('--original_image_path', type=str, help='Path to the original image.')
    parser.add_argument('--output_folder_revert', type=str, help='Path to save the reverted segmentation.')

    args = parser.parse_args()

    # args.input_folder = '/mnt/data/shared/hecktor2022/KM_Forskning_nii'
    # args.input_label_folder = '/mnt/data/shared/hecktor2022/KM_Forskning_nii'
    # args.output_folder = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/resampled'
    # args.bounding_boxes_file = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/bbox.csv'
    # args.output_folder_revert = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/revert_resample'
    # args.segmentation_path = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/unetr_pp'
    # args.original_image_path = '/mnt/data/shared/hecktor2022/KM_Forskning_nii'


    if args.mode == 'resample':
        resample_images(args.input_folder, args.input_label_folder, args.output_folder_resample, args.bounding_boxes_file)
    elif args.mode == 'revert':
        # Get only patients IDs present in prediction folder
        pat_list = os.listdir(args.segmentation_path)
        pat_list = [i.replace('.nii.gz', '') for i in pat_list]
        #print(pat_list)
        revert_segmentation(args.segmentation_path, args.bounding_boxes_file, args.original_image_path, args.output_folder_revert, pat_list)
    
