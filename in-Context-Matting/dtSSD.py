import os
import cv2
import numpy as np
from tqdm import tqdm

class Evaluator:
    def __init__(self, pred_dir, true_dir):
        self.pred_dir = pred_dir
        self.true_dir = true_dir
        self.results = []
        self.evaluate()

    def evaluate(self):
        pred_files = sorted([f for f in os.listdir(self.pred_dir) if not f.startswith('.')])
        true_files = sorted([f for f in os.listdir(self.true_dir) if not f.startswith('.')])
        assert len(pred_files) == len(true_files), "Number of files in prediction and ground truth directories must be the same."

        for i, (pred_file, true_file) in enumerate(tqdm(zip(pred_files, true_files), desc='Evaluating', total=len(pred_files))):
            pred_path = os.path.join(self.pred_dir, pred_file)
            true_path = os.path.join(self.true_dir, true_file)

            if i > 0:
                pred_path_prev = os.path.join(self.pred_dir, pred_files[i-1])
                true_path_prev = os.path.join(self.true_dir, true_files[i-1])
            else:
                pred_path_prev = None
                true_path_prev = None

            result = self.evaluate_image(pred_path, true_path, pred_path_prev, true_path_prev)
            self.results.append(result)

        avg_dtssd = np.mean(self.results)
        print(f"Average DTSSD: {avg_dtssd:.4f}")

    def evaluate_image(self, pred_path, true_path, pred_path_prev, true_path_prev):
        true_pha = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        pred_pha = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    
        if pred_path_prev and true_path_prev:
            true_pha_prev = cv2.imread(true_path_prev, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
            pred_pha_prev = cv2.imread(pred_path_prev, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    
            dtssd = MetricDTSSD()
            result = dtssd(pred_pha, pred_pha_prev, true_pha, true_pha_prev)
        else:
            result = 0.0
    
        return result


class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = np.sum(dtSSD) / np.prod(true_t.shape)
        dtSSD = np.sqrt(dtSSD)
        return dtSSD * 1e2

if __name__ == '__main__':
    pred_dir = "results"
    true_dir = "datasets/ICM57/alpha"
    Evaluator(pred_dir, true_dir)
