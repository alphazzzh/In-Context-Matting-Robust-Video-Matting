"""
HR (High-Resolution) evaluation. We found using numpy is very slow for high resolution, so we moved it to PyTorch using CUDA.

Note, the script only does evaluation. You will need to first inference yourself and save the results to disk
"""

import os
import cv2
import kornia
import numpy as np
import xlsxwriter
import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, pred_dir, true_dir, metrics=['pha_mad', 'pha_mse', 'pha_grad', 'pha_conn', 'pha_dtssd']):
        self.pred_dir = pred_dir
        self.true_dir = true_dir
        self.metrics = metrics
        self.init_metrics()
        self.evaluate()
        self.write_excel()

    def init_metrics(self):
        self.mad = MetricMAD()
        self.mse = MetricMSE()
        self.grad = MetricGRAD()
        self.conn = MetricCONN()
        self.dtssd = MetricDTSSD()

    def evaluate(self):
        pred_files = sorted([f for f in os.listdir(self.pred_dir) if not f.startswith('.')])
        true_files = sorted([f for f in os.listdir(self.true_dir) if not f.startswith('.')])
        assert len(pred_files) == len(true_files), "Number of files in prediction and ground truth directories must be the same."

        self.results = []
        pred_pha_tm1 = None
        true_pha_tm1 = None

        for i, (pred_file, true_file) in enumerate(tqdm(zip(pred_files, true_files), desc='Evaluating', total=len(pred_files))):
            pred_path = os.path.join(self.pred_dir, pred_file)
            true_path = os.path.join(self.true_dir, true_file)
            print(pred_path)
            result = self.evaluate_image(pred_path, true_path, pred_pha_tm1, true_pha_tm1, i)
            self.results.append(result)
            
            pred_pha_tm1 = result['pred_pha']
            true_pha_tm1 = result['true_pha']

    def evaluate_image(self, pred_path, true_path, pred_pha_tm1, true_pha_tm1, index):
        true_pha = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        pred_pha = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        if true_pha is None:
            raise ValueError(f"Cannot read true image from path: {true_path}")
        if pred_pha is None:
            raise ValueError(f"Cannot read predicted image from path: {pred_path}")

        true_pha = torch.from_numpy(true_pha).cuda(non_blocking=True).float().div_(255)
        pred_pha = torch.from_numpy(pred_pha).cuda(non_blocking=True).float().div_(255)

        metrics = {}
        if 'pha_mad' in self.metrics:
            metrics['pha_mad'] = self.mad(pred_pha, true_pha)
        if 'pha_mse' in self.metrics:
            metrics['pha_mse'] = self.mse(pred_pha, true_pha)
        if 'pha_grad' in self.metrics:
            metrics['pha_grad'] = self.grad(pred_pha, true_pha)
        if 'pha_conn' in self.metrics:
            metrics['pha_conn'] = self.conn(pred_pha, true_pha)
        if 'pha_dtssd' in self.metrics:
            if index == 0:
                metrics['pha_dtssd'] = 0
            else:
                metrics['pha_dtssd'] = self.dtssd(pred_pha, pred_pha_tm1, true_pha, true_pha_tm1)

        metrics['pred_pha'] = pred_pha
        metrics['true_pha'] = true_pha

        return metrics

    def write_excel(self):
        workbook = xlsxwriter.Workbook('evaluation_results.xlsx')
        worksheet = workbook.add_worksheet('evaluation')

        metrics = [key for key in self.results[0].keys() if key not in ('pred_pha', 'true_pha')]
        for i, metric in enumerate(metrics):
            worksheet.write(0, i, metric)

        total_results = {metric: [] for metric in metrics}

        for row, result in enumerate(self.results):
            for col, metric in enumerate(metrics):
                value = result.get(metric, None)
                if torch.is_tensor(value):
                    value = value.cpu().numpy()
                worksheet.write(row + 1, col, value)
                if value is not None:
                    total_results[metric].append(value)

        # Write average values
        avg_row = len(self.results) + 1
        worksheet.write(avg_row, 0, 'Average')
        for col, metric in enumerate(metrics):
            avg_value = np.mean(total_results[metric]) if total_results[metric] else 0
            worksheet.write(avg_row, col, avg_value)

        workbook.close()

class MetricMAD:
    def __call__(self, pred, true):
        return (pred - true).abs_().mean() * 1e3

class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3

class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        self.filter_x = torch.from_numpy(self.filter_x).unsqueeze(0).cuda()
        self.filter_y = torch.from_numpy(self.filter_y).unsqueeze(0).cuda()
    
    def __call__(self, pred, true):
        true_grad = self.gauss_gradient(true)
        pred_grad = self.gauss_gradient(pred)
        return ((true_grad - pred_grad) ** 2).sum() / 1000
    
    def gauss_gradient(self, img):
        img_filtered_x = kornia.filters.filter2D(img[None, None, :, :], self.filter_x, border_type='replicate')[0, 0]
        img_filtered_y = kornia.filters.filter2D(img[None, None, :, :], self.filter_y, border_type='replicate')[0, 0]
        return (img_filtered_x**2 + img_filtered_y**2).sqrt()
    
    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(j - half_size, sigma)

        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y
        
    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma**2

class MetricCONN:
    def __call__(self, pred, true):
        return (pred - true).abs_().mean() * 1e3  # Placeholder implementation

class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = dtSSD.sum() / true_t.numel()
        dtSSD = dtSSD.sqrt()
        return dtSSD * 1e2


if __name__ == '__main__':
    pred_dir = "robust"
    true_dir = "alpha"
    Evaluator(pred_dir, true_dir)
