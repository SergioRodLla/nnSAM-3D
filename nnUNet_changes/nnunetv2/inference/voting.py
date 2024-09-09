import os
import numpy as np
import SimpleITK as sitk


def save_image(array, reference_image, output_path):
    img = sitk.GetImageFromArray(array)
    img.CopyInformation(reference_image)
    sitk.WriteImage(img, output_path)

class Voting:
    def __init__(self, model_path: str, output_path: str, num_classes: int = 3):
        self.model_path = model_path
        self.output_path = output_path
        self.num_classes = num_classes
        self.models = sorted([d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d)) and d.startswith('EnableDropout_')])

    def hard_voting(self):
        output_path = os.path.join(self.output_path, 'EnableDropout_hard_voted')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

        for patient_id in patient_ids:
            if patient_id.endswith(".nii.gz"):
                seg = None
                for model in self.models:
                    pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.model_path, model, patient_id)))
                    if seg is None:
                        seg = np.zeros((self.num_classes,) + pred.shape, dtype=np.float32)
                    for class_idx in range(self.num_classes):
                        seg[class_idx] += (pred == class_idx)

                hard_voted = np.argmax(seg, axis=0)
                save_image(hard_voted, sitk.ReadImage(os.path.join(self.model_path, self.models[0], patient_id)), os.path.join(output_path, patient_id))
        print(f"Hard voting done.")

    def soft_voting(self):
        output_path = os.path.join(self.output_path, 'EnableDropout_soft_voted')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

        for patient_id in patient_ids:
            if patient_id.endswith(".nii.gz"):
                prob_sum = None
                for model in self.models:
                    pred = np.load(os.path.join(self.model_path, model, patient_id.replace('.nii.gz', '_logits.npz')))['arr_0']
                    if prob_sum is None:
                        prob_sum = pred
                    else:
                        prob_sum += pred
                
                mean_prob = prob_sum / len(self.models)
                soft_voted = np.argmax(mean_prob, axis=0)
                save_image(soft_voted, sitk.ReadImage(os.path.join(self.model_path, self.models[0], patient_id)), os.path.join(output_path, patient_id))
        print(f"Soft voting done.")

    def high_certainty(self, certainty=0.1, text='high_certainty'):
        output_path = os.path.join(self.output_path, 'EnableDropout_'+text)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

        std_devs = [] # save the sd in this list to inspect them later
        
        for patient_id in patient_ids:
            if patient_id.endswith(".npz"):
                seg = None
                for index, model in enumerate(self.models):
                    if seg is None:
                        shape = np.load(os.path.join(self.model_path, model, patient_id))['arr_0'].shape
                        seg = np.empty((len(self.models), shape[0], shape[1], shape[2], shape[3]))
                    seg[index] = np.load(os.path.join(self.model_path, model, patient_id))['arr_0']
                
                mean_seg = np.mean(seg, 0)
                mean_std_dev = np.std(seg, 0)

                # For three classes, segmentation is determined by the max probability class
                segmentation = np.argmax(mean_seg, axis=0)

                # Compute std_dev for each class
                std_dev_classes = [mean_std_dev[c][segmentation == c] for c in range(self.num_classes)]
                std_dev = np.zeros_like(segmentation, dtype=np.float32)
                for c in range(self.num_classes):
                    std_dev[segmentation == c] = std_dev_classes[c]
                
                # Apply the certainty threshold
                high_certainty = segmentation * (std_dev < certainty).astype(np.int8)
                unique_values = np.unique(high_certainty)

                save_image(high_certainty, sitk.ReadImage(os.path.join(self.model_path, self.models[0], patient_id.replace('_logits.npz', '.nii.gz'))), os.path.join(output_path, patient_id.replace('_logits.npz', '.nii.gz')))
                print(f"Uncertainty-aware voting for {patient_id.replace('_logits.npz', '')} had unique values: {unique_values}")

                # save sds for foreground classes
                foreground_std_dev = std_dev[(segmentation == 1) | (segmentation == 2)]
                std_devs.append(foreground_std_dev)
                print(f"Standard Deviation Values (Sample): {foreground_std_dev[:10]}")

        np.savez_compressed(os.path.join(output_path, 'std_devs.npz'), std_devs=np.concatenate(std_devs))
        print(f"Saved standard deviation values for inspection.")
    
    def get_class_average(self):
        # Get class averages for each patient out of the 50x3 outcomes from MCD

        output_path = os.path.join(self.output_path, 'Class_avgs')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        patient_files = os.listdir(os.path.join(self.model_path, self.models[0]))
        
        # list of patient ids
        patient_ids = [file.replace("_logits.npz", "") for file in patient_files if file.endswith("_logits.npz")]

        # dict of class averages per patient
        class_avgs = {}

        for patient_id in patient_ids:
            seg = None
            for index, model in enumerate(self.models):
                if seg is None:
                    shape = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0'].shape
                    seg = np.empty((len(self.models), shape[0], shape[1], shape[2], shape[3]))
                seg[index] = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0']
                # seg has shape (models, num_class, x, y, z)
            
            #models, num_class, x, y, z = seg.shape
            #reshape_seg = seg.reshape(models, num_class, x*y*z) # flatten x, y, z dims
            mean_seg = np.mean(seg, axis=(0, 2, 3, 4)) # compute mean across the models and dims
            class_avgs[patient_id] = mean_seg
            print(f"Class avgs done for {patient_id}")

        print(f"Total cases {len(class_avgs)}")
        np.savez_compressed(os.path.join(output_path, 'MCD_avg_class_prob_per_patient.npz'), **class_avgs)

    def get_class_average_max_prob(self):
        # Get class averages for each patient out of the 50x3 outcomes from MCD
        output_path = os.path.join(self.output_path, 'Class_avgs_max_prob')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        patient_files = os.listdir(os.path.join(self.model_path, self.models[0]))

        # list of patient ids
        patient_ids = [file.replace("_logits.npz", "") for file in patient_files if file.endswith("_logits.npz")]

        # dict of class averages per patient
        class_avgs = {}

        for patient_id in patient_ids:
            seg = None
            for index, model in enumerate(self.models):
                if seg is None:
                    shape = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0'].shape
                    seg = np.empty((len(self.models), shape[0], shape[1], shape[2], shape[3]))
                seg[index] = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0']
                # seg has shape (models, num_class, x, y, z)

            # Initialize arrays to store the maximum probabilities for each class
            max_probs = np.zeros(seg.shape[1])
            class_pixel_counts = np.zeros(seg.shape[1])

            # Iterate through each model and calculate max probabilities
            for model_index in range(seg.shape[0]):
                max_class_probs = np.max(seg[model_index], axis=0)  # Get max probs along class axis
                max_class_indices = np.argmax(seg[model_index], axis=0)  # Get indices of max probs

                for c in range(seg.shape[1]):
                    max_probs[c] += np.sum(max_class_probs[max_class_indices == c])
                    class_pixel_counts[c] += np.sum(max_class_indices == c)

            # Normalize probabilities by the number of pixels assigned to each class
            class_avgs[patient_id] = max_probs / class_pixel_counts

            print(f"Class avgs done for {patient_id} with values {class_avgs[patient_id]}")

        print(f"Total cases {len(class_avgs)}")
        np.savez_compressed(os.path.join(output_path, 'MCD_avg_max_class_prob_per_patient.npz'), **class_avgs)

    def get_patient_probs(self, patient_id):
        output_path = os.path.join(self.output_path, 'Patient_probs')
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        seg = None
        for index, model in enumerate(self.models):
            if seg is None:
                shape = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0'].shape
                seg = np.empty((len(self.models), shape[0], shape[1], shape[2], shape[3]))
            seg[index] = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0']
            # seg has shape (models, num_class, x, y, z)
        
        mean_seg = np.mean(seg, axis=0) # compute mean across the models
        print(f"Average probabilities for {patient_id} with shape {mean_seg.shape}")
        np.savez_compressed(os.path.join(output_path, patient_id + '_avg_class_prob.npz'), avg_probs=mean_seg)

    def get_combined_foreground_class_probs(self):
        output_path = os.path.join(self.output_path, 'Combined_Foreground_probs')
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        patient_files = os.listdir(os.path.join(self.model_path, self.models[0]))
        patient_ids = [file.replace("_logits.npz", "") for file in patient_files if file.endswith("_logits.npz")]

        combined_class_1_probs = []
        combined_class_2_probs = []

        for patient_id in patient_ids:
            seg = None
            for index, model in enumerate(self.models):
                if seg is None:
                    shape = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0'].shape
                    seg = np.empty((len(self.models), shape[0], shape[1], shape[2], shape[3]))
                seg[index] = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0']
                # seg has shape (models, num_class, x, y, z)

            # Average the probabilities across models
            mean_seg = np.mean(seg, axis=0)  # shape (num_class, x, y, z)

            # Determine the predicted class per pixel by taking the argmax across the class dimension
            max_class_indices = np.argmax(mean_seg, axis=0)

            # Append the averaged probabilities for pixels where the prediction is class 1 or 2
            combined_class_1_probs.extend(mean_seg[1][max_class_indices == 1].flatten())
            combined_class_2_probs.extend(mean_seg[2][max_class_indices == 2].flatten())

            print(f"Processed probabilities for {patient_id}")

        # Save the combined probabilities for inspection
        np.savez_compressed(os.path.join(output_path, 'combined_class_1_probs.npz'), class_1_probs=np.array(combined_class_1_probs))
        np.savez_compressed(os.path.join(output_path, 'combined_class_2_probs.npz'), class_2_probs=np.array(combined_class_2_probs))

        print("Saved combined class 1 and class 2 probabilities for all patients.")

    def get_combined_foreground_class_probs_and_std(self):
        output_path = os.path.join(self.output_path, 'Combined_Foreground_probs_and_std')
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        patient_files = os.listdir(os.path.join(self.model_path, self.models[0]))
        patient_ids = [file.replace("_logits.npz", "") for file in patient_files if file.endswith("_logits.npz")]

        combined_class_1_probs = []
        combined_class_2_probs = []
        combined_class_1_std = []
        combined_class_2_std = []

        for patient_id in patient_ids:
            seg = None
            for index, model in enumerate(self.models):
                if seg is None:
                    shape = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0'].shape
                    seg = np.empty((len(self.models), shape[0], shape[1], shape[2], shape[3]))
                seg[index] = np.load(os.path.join(self.model_path, model, patient_id + "_logits.npz"))['arr_0']
                # seg has shape (models, num_class, x, y, z)

            # Average and standard deviation of probabilities across models
            mean_seg = np.mean(seg, axis=0)  # shape (num_class, x, y, z)
            std_seg = np.std(seg, axis=0)    # shape (num_class, x, y, z)

            # Determine the predicted class per pixel by taking the argmax across the class dimension
            max_class_indices = np.argmax(mean_seg, axis=0)

            # Append the averaged probabilities and standard deviations for pixels where the prediction is class 1 or 2
            combined_class_1_probs.extend(mean_seg[1][max_class_indices == 1].flatten())
            combined_class_2_probs.extend(mean_seg[2][max_class_indices == 2].flatten())
            combined_class_1_std.extend(std_seg[1][max_class_indices == 1].flatten())
            combined_class_2_std.extend(std_seg[2][max_class_indices == 2].flatten())

            print(f"Processed probabilities and standard deviations for {patient_id}")

        # Save the combined probabilities and standard deviations for inspection
        np.savez_compressed(os.path.join(output_path, 'combined_class_1_probs_and_std.npz'),
                            class_1_probs=np.array(combined_class_1_probs),
                            class_1_std=np.array(combined_class_1_std))
        np.savez_compressed(os.path.join(output_path, 'combined_class_2_probs_and_std.npz'),
                            class_2_probs=np.array(combined_class_2_probs),
                            class_2_std=np.array(combined_class_2_std))

        print("Saved combined class 1 and class 2 probabilities and standard deviations for all patients.")

    def soft_voting_with_class_specific_thresholds(self, outdir_name, GTVp_avg_thr, GTVn_avg_thr):
        output_path = os.path.join(self.output_path, 'Prob_avg_thr_soft_voting_'+outdir_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

        for patient_id in patient_ids:
            if patient_id.endswith(".nii.gz"):
                prob_sum = None
                for model in self.models:
                    pred = np.load(os.path.join(self.model_path, model, patient_id.replace('.nii.gz', '_logits.npz')))['arr_0']
                    if prob_sum is None:
                        prob_sum = pred
                    else:
                        prob_sum += pred
                
                mean_prob = prob_sum / len(self.models)
            
                soft_voting = np.argmax(mean_prob, axis=0)
                pixel_class_prob = np.max(mean_prob, axis=0)

                # see if pixel class probabilty is below respective threshold
                GTVp_exclude = (soft_voting == 1) & (pixel_class_prob < GTVp_avg_thr)
                GTVn_exclude = (soft_voting == 2) & (pixel_class_prob < GTVn_avg_thr)

                # set previous pixel's class as background
                soft_voting[GTVp_exclude] = 0
                soft_voting[GTVn_exclude] = 0

                save_image(soft_voting, sitk.ReadImage(os.path.join(self.model_path, self.models[0], patient_id)), os.path.join(output_path, patient_id))
                print(f"Soft voting done for patient: {patient_id}")
        
        print(f"Soft voting with {outdir_name} prob. avg. thresholds DONE")

    def uncertainty_voting_avg_and_std_thr(self, outdir_name, GTVp_avg_thr, GTVn_avg_thr, GTVp_std_thr, GTVn_std_thr):
        output_path = os.path.join(self.output_path, 'Uncertainty_voting_avg_std_thr_'+outdir_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

        # for patient_id in patient_ids:
        #     if patient_id.endswith(".nii.gz"):
        #         prob_sum = None
        #         prob_squares_sum = None
        #         for model in self.models:
        #             pred = np.load(os.path.join(self.model_path, model, patient_id.replace('.nii.gz', '_logits.npz')))['arr_0']
        #             if prob_sum is None:
        #                 prob_sum = pred
        #                 prob_squares_sum = pred ** 2
        #             else:
        #                 prob_sum += pred
        #                 prob_squares_sum += pred ** 2
                
        #         num_models = len(self.models)
        #         mean_prob = prob_sum / num_models
        #         epsilon = 1e-6  # A small value to ensure non-negative inside sqrt
        #         std_prob = np.sqrt(np.maximum(prob_squares_sum / num_models - (mean_prob ** 2), 0) + epsilon)

        for patient_id in patient_ids:
            if patient_id.endswith(".nii.gz"):
                prob_list = []

                for model in self.models:
                    pred = np.load(os.path.join(self.model_path, model, patient_id.replace('.nii.gz', '_logits.npz')))['arr_0']
                    prob_list.append(pred)

                # Stack all predictions to compute mean and std
                stacked_probs = np.stack(prob_list)
                mean_prob = np.mean(stacked_probs, axis=0)
                std_prob = np.std(stacked_probs, axis=0)

                # Initial soft voting without threshold
                soft_voted = np.argmax(mean_prob, axis=0)
                
                # Thresholding for class 1 and class 2 based on probabilities
                max_probs = np.max(mean_prob, axis=0)
                class_1_mask = (soft_voted == 1) & (max_probs < GTVp_avg_thr)
                class_2_mask = (soft_voted == 2) & (max_probs < GTVn_avg_thr)
                
                # Set to background if below the respective probability threshold
                soft_voted[class_1_mask] = 0
                soft_voted[class_2_mask] = 0

                # Thresholding based on standard deviation
                std_class_1_mask = (soft_voted == 1) & (std_prob[1] > GTVp_std_thr)
                std_class_2_mask = (soft_voted == 2) & (std_prob[2] > GTVn_std_thr)

                # Set to background if above the respective standard deviation threshold
                soft_voted[std_class_1_mask] = 0
                soft_voted[std_class_2_mask] = 0

                save_image(soft_voted, sitk.ReadImage(os.path.join(self.model_path, self.models[0], patient_id)), os.path.join(output_path, patient_id))
                print(f"Uncertainty voting done for patient: {patient_id}")

        print(f"Uncertainty voting with {outdir_name} thresholds DONE")


def main(): 
    model_path = '/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/test_preds_MCD'
    output_path = '/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/test_preds_MCD/voting_testSTD'
    voting = Voting(model_path, output_path)
    # voting.hard_voting()
    # voting.soft_voting()
    # print("Soft voting DONE")
    # voting.high_certainty(certainty=0.1, text='certainty_01')
    # print("certainty 0.1 DONE")
    #voting.high_certainty(certainty=0.02, text='certainty_2e-2')
    #print("certainty 0.02 DONE")
    # voting.high_certainty(certainty=0.02, text='high_certainty')
    # print("certainty high certainty DONE")
    # voting.high_certainty(certainty=0.05, text='moderate_certainty')
    # print("certainty moderate certainty DONE")
    # voting.high_certainty(certainty=0.1, text='low_certainty')
    # print("certainty low certainty percentile")
    # voting.high_certainty(certainty=0.15, text='certainty_15e-2')
    # print("certainty_15e-2 DONE")
    # voting.high_certainty(certainty=0.2, text='certainty_2e-1')
    # print("certainty_2e-1 DONE")
    # voting.get_class_average()
    # print("Class avgs finished!!")
    # voting.get_class_average_max_prob()
    # print("Class avgs max. prob. finished!!")
    # voting.get_patient_probs(patient_id="CHUP_070")
    # voting.get_combined_foreground_class_probs_and_std()
    # voting.soft_voting_with_class_specific_thresholds(outdir_name="1.5th_p", GTVp_avg_thr=0.58, GTVn_avg_thr=0.56)
    # voting.soft_voting_with_class_specific_thresholds(outdir_name="2.5th_p", GTVp_avg_thr=0.64, GTVn_avg_thr=0.6)
    # voting.soft_voting_with_class_specific_thresholds(outdir_name="3.5th_p", GTVp_avg_thr=0.69, GTVn_avg_thr=0.63)

    # Uncertainty with AVG and STD
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="2.5p_AVG_96.5p_STD", 
    #                                           GTVp_avg_thr=0.64, GTVn_avg_thr=0.6, 
    #                                           GTVp_std_thr=0.138, GTVn_std_thr=0.155)
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="2.5p_AVG_97.5p_STD", 
    #                                           GTVp_avg_thr=0.64, GTVn_avg_thr=0.6, 
    #                                           GTVp_std_thr=0.158, GTVn_std_thr=0.173)
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="2.5p_AVG_98.5p_STD", 
    #                                           GTVp_avg_thr=0.64, GTVn_avg_thr=0.6, 
    #                                           GTVp_std_thr=0.186, GTVn_std_thr=0.196)
    
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="5p_AVG_96.5p_STD", 
    #                                           GTVp_avg_thr=0.76, GTVn_avg_thr=0.69, 
    #                                           GTVp_std_thr=0.138, GTVn_std_thr=0.155)
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="5p_AVG_97.5p_STD", 
    #                                           GTVp_avg_thr=0.76, GTVn_avg_thr=0.69, 
    #                                           GTVp_std_thr=0.158, GTVn_std_thr=0.173)
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="5p_AVG_98.5p_STD", 
    #                                           GTVp_avg_thr=0.76, GTVn_avg_thr=0.69, 
    #                                           GTVp_std_thr=0.186, GTVn_std_thr=0.196)
    
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="10p_AVG_96.5p_STD", 
    #                                           GTVp_avg_thr=0.91, GTVn_avg_thr=0.83, 
    #                                           GTVp_std_thr=0.138, GTVn_std_thr=0.155)
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="10p_AVG_97.5p_STD", 
    #                                           GTVp_avg_thr=0.91, GTVn_avg_thr=0.83, 
    #                                           GTVp_std_thr=0.158, GTVn_std_thr=0.173)
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="10p_AVG_98.5p_STD", 
    #                                           GTVp_avg_thr=0.91, GTVn_avg_thr=0.83, 
    #                                           GTVp_std_thr=0.186, GTVn_std_thr=0.196)
    
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="2.5p_AVG_99p_STD", 
    #                                           GTVp_avg_thr=0.64, GTVn_avg_thr=0.6, 
    #                                           GTVp_std_thr=0.205, GTVn_std_thr=0.213)
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="2.5p_AVG_99.5p_STD", 
    #                                           GTVp_avg_thr=0.64, GTVn_avg_thr=0.6, 
    #                                           GTVp_std_thr=0.235, GTVn_std_thr=0.24)

    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="0.5p_AVG_99p_STD", 
    #                                           GTVp_avg_thr=0.527, GTVn_avg_thr=0.519, 
    #                                           GTVp_std_thr=0.205, GTVn_std_thr=0.213)
    # voting.uncertainty_voting_avg_and_std_thr(outdir_name="1p_AVG_99.5p_STD", 
    #                                           GTVp_avg_thr=0.555, GTVn_avg_thr=0.539, 
    #                                           GTVp_std_thr=0.235, GTVn_std_thr=0.24)
    voting.uncertainty_voting_avg_and_std_thr(outdir_name="zero_AVG_99.5p_STD", 
                                              GTVp_avg_thr=0, GTVn_avg_thr=0, 
                                              GTVp_std_thr=0.235, GTVn_std_thr=0.24)
    
    voting.uncertainty_voting_avg_and_std_thr(outdir_name="zero_AVG_97.5p_STD", 
                                              GTVp_avg_thr=0, GTVn_avg_thr=0, 
                                              GTVp_std_thr=0.158, GTVn_std_thr=0.173)
    
    voting.uncertainty_voting_avg_and_std_thr(outdir_name="zero_AVG_98.5p_STD", 
                                              GTVp_avg_thr=0, GTVn_avg_thr=0, 
                                              GTVp_std_thr=0.186, GTVn_std_thr=0.196)

    
if __name__ == "__main__":
    main()
