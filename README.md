## 快速体验
radxa 提供一个开箱即用的 YOLOv4 目标检测例子，旨在用户可以直接在 sirider s1 使用 AIPU 推理 yolov4_tiny 模型，
免去复杂的模型编译和执行代码编译， 这对想快速使用 AIPU 而不想从头编译模型的用户是最佳的选择，
如您对完整工作流程感兴趣可以参考 [详细教程](#详细教程) 章节。

- 克隆仓库代码
    ```bash
    git clone https://github.com/zifeng-radxa/siriders1_NPU_yolov4_tiny_demo.git
    ```

- 安装依赖
    :::tip
    建议使用 virtualenv
    :::
    
    ```bash
    cd siriders1_NPU_yolov4_tiny_demo
    pip3 install -r requirements.txt
    ```

- 运行 yolov4 demo 程序
    ```bash
    python3 yolov4_aipu.py -m [mode] -i [your_input_path] -r 
    ```
    参数解析:
    
        `-h`, `--help`: 打印参数信息
 
        `-m`, `--mode`: 输入模式选择，支持['camera', 'video', 'image']

        `-i`, `--input`: 输入文件路径, 当 mode 为 ['video', 'image'] 时请提供文件路径 

        `-r`, `--real_time`: 实时预览

        `-s`, `--save`: 保存输出，保存在 `output` 文件夹
