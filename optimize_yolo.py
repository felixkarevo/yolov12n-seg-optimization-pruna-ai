import torch
import time # Import time for benchmarking
from ultralytics import YOLO  # Assuming you use ultralytics YOLO, adjust if needed

from pruna import smash, SmashConfig
from pruna.logging.logger import PrunaLoggerContext

# --- Configuration ---
MODEL_PATH = "models/yolov12n-seg-residual.pt" # Make sure this path is correct
SMASHED_MODEL_PATH = "models/yolov12n-seg-residual_smashed_tc_gpu.pt" # Path to save the optimized model

# Benchmarking parameters
NUM_WARMUP_RUNS = 10
NUM_TIMED_RUNS = 30
# IMPORTANT: Adjust the input shape if needed for your specific YOLO model
SAMPLE_INPUT_SHAPE = (1, 3, 640, 640)

# --- Helper Function for Benchmarking ---
def benchmark_inference(model_part, sample_input, device):
    """Runs inference multiple times and returns average latency."""
    model_part.eval() # Ensure model is in eval mode
    model_part.to(device)
    sample_input = sample_input.to(device)

    latencies = []
    with torch.no_grad():
        # Warm-up runs
        print(f"  Performing {NUM_WARMUP_RUNS} warm-up runs...")
        for _ in range(NUM_WARMUP_RUNS):
            _ = model_part(sample_input)
        # Ensure CUDA synchronization if on GPU
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        print(f"  Performing {NUM_TIMED_RUNS} timed runs...")
        for _ in range(NUM_TIMED_RUNS):
            start_time = time.perf_counter()
            _ = model_part(sample_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    print(f"  Average latency: {avg_latency_ms:.3f} ms")
    return avg_latency_ms

# --- Pruna Optimization and Benchmarking ---

# Enable verbose logging to see Pruna's progress
with PrunaLoggerContext(verbose=True):
    # Determine device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    if device == "cpu":
        print("Warning: Running on CPU. GPU optimization requested but no CUDA device found.")

    # 1. Load your YOLO model
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model_to_smash = model.model # Common for ultralytics YOLOv8
        print("Model loaded successfully.")

        # Create sample input tensor
        # Use float() for standard models, adjust dtype if needed (e.g., half() for FP16)
        sample_input = torch.randn(SAMPLE_INPUT_SHAPE).float()

        # --- Benchmark Before Optimization ---
        print("\nBenchmarking ORIGINAL model...")
        original_latency = benchmark_inference(model_to_smash, sample_input, device)

    except Exception as e:
        print(f"Error loading model or during initial benchmark: {e}")
        print("Please ensure the model path, input shape, and necessary libraries are correct.")
        exit()

    # 2. Configure Pruna SmashConfig
    smash_config = SmashConfig()
    smash_config["compiler"] = "torch_compile"
    smash_config["device"] = device # Inform Pruna about the target device

    # --- Force GPU backend for torch_compile ---
    # Set backend to 'inductor' for NVIDIA GPU optimization
    # Hyperparameters are typically prefixed with the algorithm name
    smash_config["torch_compile_backend"] = "inductor"
    # Optional: You might want max-autotune for potentially more speed at the cost of compile time
    # smash_config["torch_compile_mode"] = "max-autotune"
    print(f"Configured torch_compile backend: {smash_config['torch_compile_backend']}") # Read the set value back

    # 3. Apply Pruna optimization
    print("\nStarting Pruna smash process...")
    try:
        smashed_model_part_wrapper = smash(model=model_to_smash, smash_config=smash_config)
        print("Pruna smash process completed successfully.")

        # Extract the actual optimized model module
        smashed_model_part = smashed_model_part_wrapper.model

        # --- Benchmark After Optimization ---
        print("\nBenchmarking SMASHED model...")
        smashed_latency = benchmark_inference(smashed_model_part, sample_input, device)

        # --- Print Results ---
        print("\n--- Benchmarking Results ---")
        print(f"Original model average latency: {original_latency:.3f} ms")
        print(f"Smashed model average latency:  {smashed_latency:.3f} ms")
        if original_latency > 0:
            speedup = original_latency / smashed_latency
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("Could not calculate speedup (original latency was zero).")
        print("---------------------------")

        # 4. Replace the original model part with the smashed one
        model.model = smashed_model_part

        # 5. Save the smashed model
        print(f"\nSaving smashed model to: {SMASHED_MODEL_PATH}")
        # For ultralytics YOLO:
        model.save(SMASHED_MODEL_PATH)
        # For raw PyTorch models (alternative):
        # torch.save(smashed_model_part.state_dict(), SMASHED_MODEL_PATH) # Save state_dict
        print("Smashed model saved successfully.")

        print("\nModel is optimized, benchmarked, saved, and ready for inference.")

    except Exception as e:
        print(f"\nError during Pruna smash process or final benchmarking/saving: {e}")

print("Script finished.") 