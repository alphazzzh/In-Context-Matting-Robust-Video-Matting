import os
import cv2
import numpy as np
import xlsxwriter
from tqdm import tqdm


class Evaluator:
    def __init__(self, pred_dir, true_dir, metrics=['pha_mad', 'pha_mse', 'pha_grad', 'pha_conn', 'fgr_mad', 'fgr_mse']):
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
        
    def evaluate(self):
        self.results = []
        pred_files = sorted(os.listdir(self.pred_dir))
        true_files = sorted(os.listdir(self.true_dir))
        for pred_file, true_file in tqdm(zip(pred_files, true_files), total=len(pred_files)):
            pred_path = os.path.join(self.pred_dir, pred_file)
            true_path = os.path.join(self.true_dir, true_file)
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            true_img = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            metrics = {}
            for metric_name in self.metrics:
                metric_func = getattr(self, metric_name)
                metrics[metric_name] = metric_func(pred_img, true_img)
            self.results.append((pred_file, metrics))
        
    def write_excel(self):
        workbook = xlsxwriter.Workbook(os.path.join(self.pred_dir, f'{os.path.basename(self.pred_dir)}.xlsx'))
        metricsheets = [workbook.add_worksheet(metric) for metric in self.metrics]
             
        for row, (pred_file, metrics) in enumerate(self.results):
            for metricsheet, metric_name in zip(metricsheets, self.metrics):
                # Write the header
                if row == 0:
                    metricsheet.write(1, 0, 'Average')
                    metricsheet.write(1, 1, f'=AVERAGE(B3:B17)')
                    
                metricsheet.write(row + 2, 0, pred_file)
                metricsheet.write(row + 2, 1, metrics[metric_name])
        
        workbook.close()


    def pha_mad(self, pred, true):
        return self.mad(pred, true)

    def pha_mse(self, pred, true):
        return self.mse(pred, true)

    def pha_grad(self, pred, true):
        return self.grad(pred, true)

    def pha_conn(self, pred, true):
        return self.conn(pred, true)
    
    def fgr_mad(self, pred, true):
        return self.mad(pred, true)
    
    def fgr_mse(self, pred, true):
        return self.mse(pred, true)


class MetricMAD:
    def __call__(self, pred, true):
        return np.abs(pred - true).mean() * 1e3


class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3


class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
    
    def __call__(self, pred, true):
        pred_normed = np.zeros_like(pred)
        true_normed = np.zeros_like(true)
        cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(true, true_normed, 1., 0., cv2.NORM_MINMAX)

        true_grad = self.gauss_gradient(true_normed).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed).astype(np.float32)

        grad_loss = ((true_grad - pred_grad) ** 2).sum()
        return grad_loss / 1000
    
    def gauss_gradient(self, img):
        img_filtered_x = cv2.filter2D(img, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(img, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
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
        step=0.1
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(true)
        for i in range(1, len(thresh_steps)):
            true_thresh = true >= thresh_steps[i]
            pred_thresh = pred >= thresh_steps[i]
            intersection = (true_thresh & pred_thresh).astype(np.uint8)

            # connected components
            _, output, stats, _ = cv2.connectedComponentsWithStats(
                intersection, connectivity=4)
            # start from 1 in dim 0 to exclude background
            size = stats[1:, -1]

            # largest connected component of the intersection
            omega = np.zeros_like(true)
            if len(size) != 0:
                max_id = np.argmax(size)
                # plus one to include background
                omega[output == max_id + 1] = 1

            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]
        round_down_map[round_down_map == -1] = 1

        true_diff = true - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        true_phi = 1 - true_diff * (true_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

        connectivity_error = np.sum(np.abs(true_phi - pred_phi))
        return connectivity_error / 1000



if __name__ == '__main__':
    pred_dir = "robust"
    true_dir = "alpha"
    Evaluator(pred_dir, true_dir)
