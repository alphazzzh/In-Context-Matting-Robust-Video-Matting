# in-Context-Matting 代码运行说明

## 需求
- 需要下载模型 [Stable Diffusion Version 2](https://github.com/Stability-AI/StableDiffusion#requirements). 存放路径为./stabilityai/stable。
- 从 [链接](https://pan.baidu.com/s/1HPbRRE5ZtPRpOSocm9qOmA?pwd=BA1c)下载预训练模型。


## 用法

   1. 使用以下命令运行预测脚本。如果文件存放路径不同，请用实际路径替换它们。
   ```bash
   python eval.py --checkpoint PATH_TO_MODEL --save_path results/ --config config/eval.yaml
   ```
   2. 预测脚本完成时会附带MSE、SAD、GRAD、CONN四个指标的信息，要获得dtSSD指标，在终端中运行以下代码。

   ```bash
   python dtSSD.py
   ```
   
   
## 数据集
数据集已经包含在库中，无需准备和修改。
