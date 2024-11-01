## Quick Start
Radxa provides a ready-to-use YOLOv4 object detection example that allows users to run the yolov4_tiny model with AIPU on the Sirider S1 directly. This example eliminates the need for complex model and execution code compilation, making it ideal for users who want to use the AIPU quickly without compiling from scratch. 

- Clone the repository:
    ```bash
    git clone https://github.com/zifeng-radxa/siriders1_NPU_yolov4_tiny_demo.git
    ```

- Install dependencies:
    :::tip
    It is recommended to use virtualenv.
    :::
    
    ```bash
    cd siriders1_NPU_yolov4_tiny_demo/demo
    pip3 install -r requirements.txt
    ```

- Run the yolov4 demo program:
    ```bash
    python3 yolov4_aipu.py -m [mode] -i [your_input_path] -r 
    ```
    Parameters:

        `-h`, `--help`: Print parameter information.
 
        `-m`, `--mode`: Input mode selection, supports ['camera', 'video', 'image'].

        `-i`, `--input`: Input file path; please provide the file path when mode is set to ['video', 'image'].

        `-r`, `--real_time`: Real-time preview.

        `-s`, `--save`: Save output to the `output` folder.
