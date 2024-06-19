import torch
from model import MattingNetwork
from inference import convert_video

model = MattingNetwork('mobilenetv3').eval().cuda() 
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))


convert_video(
    model,                           # 模型，可以加载到任何设备（cpu 或 cuda）
    input_source='2.mp4',        # 视频文件，或图片序列文件夹
    output_type='video',             # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
    output_composition='com.mp4',    # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
    output_alpha="pha.mp4",          # [可选项] 输出透明度预测
    output_foreground="fgr.mp4",     # [可选项] 输出前景预测
    output_video_mbps=4,             # 若导出视频，提供视频码率
    downsample_ratio=None,           # 下采样比，可根据具体视频调节，或 None 选择自动
    seq_chunk=12, 
)
