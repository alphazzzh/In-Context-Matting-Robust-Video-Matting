# RobustVideoMatting 代码运行说明



## 用法说明
1. 预训练模型和数据集已经包含在库中。

2. 要获得预测视频，执行以下代码。
   ```bash
   python useit.py
   ```

3. 计算各指标，执行以下代码（evaluation中已经包含了经过PS处理后的结果）。
   ```bash
   cd evaluation
   python evaluate_hr.py
   ```