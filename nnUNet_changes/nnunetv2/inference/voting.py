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
        self.models = sorted([d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d)) and d.startswith('model_')])

    def hard_voting(self):
        output_path = os.path.join(self.output_path, 'hard_voted')
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
        output_path = os.path.join(self.output_path, 'soft_voted')
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
        output_path = os.path.join(self.output_path, text)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        patient_ids = os.listdir(os.path.join(self.model_path, self.models[0]))

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

                # segmentation is determined by the max probability class
                segmentation = np.argmax(mean_seg, axis=0)

                # get std_dev for each class
                std_dev_classes = [mean_std_dev[c][segmentation == c] for c in range(self.num_classes)]
                std_dev = np.zeros_like(segmentation, dtype=np.float32)
                for c in range(self.num_classes):
                    std_dev[segmentation == c] = std_dev_classes[c]
                
                # certainty threshold
                high_certainty = segmentation * (std_dev < certainty).astype(np.int8)
                unique_values = np.unique(high_certainty)

                save_image(high_certainty, sitk.ReadImage(os.path.join(self.model_path, self.models[0], patient_id.replace('_logits.npz', '.nii.gz'))), os.path.join(output_path, patient_id.replace('_logits.npz', '.nii.gz')))
                print(f"Uncertainty-aware voting for {patient_id.replace('_logits.npz', '')} had unique values: {unique_values}")


def main() -> None: 
    model_path = '/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/test_preds_MCD'
    output_path = '/media/HDD_4TB_2/sergio/TFM/hecktor/hecktor/test_preds_MCD/voting'
    voting = Voting(model_path, output_path)
    #voting.hard_voting()
    #voting.soft_voting()
    #voting.high_certainty(certainty=0.1, text='high_certainty')
    voting.high_certainty(certainty=0.05, text='certainty_5e-2')
    print("certainty 0.05 DONE")
    voting.high_certainty(certainty=0.02, text='certainty_2e-2')
    print("certainty 0.02 DONE")

if __name__ == "__main__":
    main()
