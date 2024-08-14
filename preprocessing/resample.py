
from pathlib import Path
from multiprocessing import Pool
import logging

import click
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os 
import sys

def main(arguments):
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
    pt = sitk.ReadImage(
        str([f for f in input_folder.rglob(p + "__PT*")][0].resolve()))
    gtvt = sitk.ReadImage(
        str([f for f in input_label_folder.rglob(p + "*")][0].resolve()))
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])  # sitk is so stupid
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct = resampler.Execute(ct)
    pt = resampler.Execute(pt)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    gtvt = resampler.Execute(gtvt)

    print(f'Patient {p} resample completed. ')
    sys.stdout.flush()

    sitk.WriteImage(ct, str(
        (output_folder / (p + "__CT.nii.gz")).resolve()))
    sitk.WriteImage(pt, str(
        (output_folder / (p + "__PT.nii.gz")).resolve()))
    sitk.WriteImage(gtvt,
                    str((output_folder / (p + "__gtv.nii.gz")).resolve()))
    print(f'Patient {p} saved.')
    sys.stdout.flush()

def main_test(arguments):
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
    p, input_folder, output_folder, bb_df = arguments
    resampling=(1, 1, 1)
    output_folder = Path(output_folder)
    input_folder = Path(input_folder)
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
    pt = sitk.ReadImage(
        str([f for f in input_folder.rglob(p + "__PT*")][0].resolve()))
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])  # sitk is so stupid
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct = resampler.Execute(ct)
    pt = resampler.Execute(pt)

    print(f'Patient {p} resample completed. ')
    sys.stdout.flush()

    sitk.WriteImage(ct, str(
        (output_folder / (p + "__CT.nii.gz")).resolve()))
    sitk.WriteImage(pt, str(
        (output_folder / (p + "__PT.nii.gz")).resolve()))
    print(f'Patient {p} saved.')
    sys.stdout.flush()

if __name__ == '__main__':
    # input_folder = '/mnt/data/shared/hecktor2022/train/hecktor2022_training/hecktor2022/imagesTr/'
    # input_label_folder = '/mnt/data/shared/hecktor2022/train/hecktor2022_training/hecktor2022/labelsTr/'
    # output_folder = '/mnt/data/shared/hecktor2022/train/hecktor2022_training/hecktor2022/resampled/'
    # bounding_boxes_file = '/mnt/data/shared/hecktor2022/train/hecktor2022_training/hecktor2022/bbox/bb_box_training.csv'
    
    # train set
    # input_folder = '/media/sergio/TOSHIBA EXT/Master/TFM/HECKTOR/hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022/imagesTr/'
    # input_label_folder = '/media/sergio/TOSHIBA EXT/Master/TFM/HECKTOR/hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022/labelsTr/'
    # output_folder = '/media/sergio/TOSHIBA EXT/Master/TFM/HECKTOR/hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022/resampled/'
    # bounding_boxes_file = '/media/sergio/TOSHIBA EXT/Master/TFM/HECKTOR/hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022/bbox/bb_box_training.csv'

    # test set
    input_folder = '/media/sergio/TOSHIBA EXT/Master/TFM/HECKTOR/hecktor2022_testing_v2/hecktor2022_testing/imagesTs/'
    output_folder = '/media/sergio/TOSHIBA EXT/Master/TFM/HECKTOR/hecktor2022_testing_v2/hecktor2022_testing/resampled/'
    bounding_boxes_file = '/media/sergio/TOSHIBA EXT/Master/TFM/HECKTOR/hecktor2022_testing_v2/hecktor2022_testing/bbox/bb_box_testing.csv'

    #cores = 96
    cores = 8

    bb_df = pd.read_csv(bounding_boxes_file)

    patient_list = sorted(list(bb_df['PatientID']))
    bb_df = bb_df.set_index('PatientID')

    # list_of_args = zip(patient_list, [input_folder]*len(patient_list),
    #                    [input_label_folder]*len(patient_list), [output_folder]*len(patient_list),
    #                      [bb_df]*len(patient_list) )
    
    list_of_args_test = zip(patient_list, [input_folder]*len(patient_list),
                       [output_folder]*len(patient_list),
                         [bb_df]*len(patient_list) )

    
    with Pool(cores) as p:
        p.map(main_test, list_of_args_test)
        #p.starmap(main, list_of_args)
    #main()