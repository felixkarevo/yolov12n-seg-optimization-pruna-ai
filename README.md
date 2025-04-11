# YOLOv12n-seg Optimization with Pruna AI

This repository contains code to optimize the YOLOv12n-seg-residual model using Pruna AI's optimization tools. The optimization process significantly improves inference speed while maintaining model accuracy.

## Background

[YOLOv12](https://github.com/sunsmarterjie/yolov12) is a state-of-the-art "Attention-Centric Real-Time Object Detector" developed by Yunjie Tian, Qixiang Ye, and David Doermann. The YOLOv12n-seg-residual variant adds instance segmentation capabilities to the base detection model. This repository specifically focuses on optimizing this model using Pruna AI's optimization tools, which leverage PyTorch's compilation capabilities to achieve significant performance improvements.

The optimization is performed using Pruna's `smash` functionality, which applies various optimization techniques including graph transformations and compilation optimizations to make the model run faster without compromising accuracy.

## Model Source

The YOLOv12n-seg-residual model used in this project was sourced from Weights & Biases:
[YOLOv12n-seg-residual Model](https://wandb.ai/laughing/YOLO12/artifacts/model/run_1mllmpe1_model/v0/files)

Note: YOLOv12 is relatively new, having been published in February 2025 on arXiv as "YOLOv12: Attention-Centric Real-Time Object Detectors."

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/yolov12n-seg-optimization-pruna-ai.git
cd yolov12n-seg-optimization-pruna-ai
```

### Set Up Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download the Model

1. Download the YOLOv12n-seg-residual model from [Weights & Biases](https://wandb.ai/laughing/YOLO12/artifacts/model/run_erwr1jwp_model/v0/files)
2. Place the downloaded `.pt` file in the `models/` directory:

```bash
# Create models directory if it doesn't exist
mkdir -p models
# Move the downloaded model to the models directory
mv path/to/downloaded/yolov12n-seg-residual.pt models/
```

### Run the Optimization Script

```bash
python optimize_yolo.py
```

This will:
1. Load the YOLOv12n-seg-residual model
2. Benchmark the original model's performance
3. Apply Pruna's optimization techniques
4. Benchmark the optimized model's performance
5. Save the optimized model to `models/yolov12n-seg-residual_smashed_tc_gpu.pt`

### Configuration Options

You can modify the following parameters in `optimize_yolo.py`:

- `MODEL_PATH`: Path to the original YOLOv12n-seg-residual model
- `SMASHED_MODEL_PATH`: Path to save the optimized model
- `NUM_WARMUP_RUNS`: Number of warm-up inference runs before benchmarking
- `NUM_TIMED_RUNS`: Number of inference runs for benchmarking
- `SAMPLE_INPUT_SHAPE`: Input shape for the model (default: [1, 3, 640, 640])

## Technical Details

### Optimization Process

The optimization process uses Pruna AI's `smash` functionality with PyTorch's compilation backend. The script:

1. Loads the YOLOv12n-seg-residual model 
2. Configures Pruna's SmashConfig with appropriate compiler settings
3. Applies PyTorch's inductor backend optimization for NVIDIA GPUs
4. Benchmarks the original and optimized models to measure performance improvement

### Performance Improvements

Typical performance improvements vary depending on hardware, but you can expect:
- 1.5-3x speedup on NVIDIA GPUs
- Improved throughput for real-time applications

### About YOLOv12

YOLOv12 is an attention-centric YOLO framework that matches the speed of CNN-based models while harnessing the performance benefits of attention mechanisms. According to the authors:

- YOLOv12-N achieves 40.6% mAP with an inference latency of 1.64 ms on a T4 GPU
- It outperforms advanced YOLOv10-N / YOLOv11-N by 2.1%/1.2% mAP with comparable speed
- The model also surpasses end-to-end real-time detectors like RT-DETR and RT-DETRv2

## Requirements

The main dependencies are:
- PyTorch (>= 2.0.0)
- Ultralytics (for the YOLO class used to load and operate the model)
- Pruna (for optimization)

See `requirements.txt` for the complete list of dependencies.

## License

MIT License

## Acknowledgments

- [YOLOv12](https://github.com/sunsmarterjie/yolov12) by Yunjie Tian, Qixiang Ye, and David Doermann
- [Pruna AI](https://github.com/pruna-ai) for their optimization tools 