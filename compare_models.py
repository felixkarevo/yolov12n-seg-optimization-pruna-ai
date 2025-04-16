import torch
import time
import os
import glob
import statistics # Import statistics for median calculation
from ultralytics import YOLO

# --- Configuration ---
MODEL_DIR = "models"
IMAGE_DIR = "valid/images"
IMAGE_PATTERN = "*.png" # Adjust if you have other image types (e.g., *.jpg)

# Model filenames
MODEL_CONFIG = {
    "yolov8n-seg": {
        "original": "yolov8n-seg-640.pt",
        "smashed": "yolov8n-seg-640smashed_tc_gpu.pt"
    },
    "yolov12n-seg-residual": {
        "original": "yolov12n-seg-residual-640.pt",
        "smashed": "yolov12n-seg-residual-640smashed_tc_gpu.pt"
    }
}

# Benchmarking parameters
NUM_WARMUP_RUNS = 100 # Number of initial runs on the first image to warm up
NUM_TIMED_RUNS = 700  # Number of timed runs on the first image after warm-up

# --- Helper Function for Benchmarking ---
def benchmark_model(model_path, image_paths, device):
    """Loads a model, runs inference, calculates mean/median latency."""
    print(f"Benchmarking {os.path.basename(model_path)} on {device}...")
    if not os.path.exists(model_path):
        print(f"  Error: Model file not found at {model_path}")
        return None
    if not image_paths:
        print("  Error: No images found for benchmarking.")
        return None

    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"  Model loaded successfully.")
    except Exception as e:
        print(f"  Error loading model {model_path}: {e}")
        return None

    inference_latencies_ms = [] # Store inference-specific latencies in ms
    first_image = image_paths[0]

    try:
        # Warm-up runs (don't need to store results)
        print(f"  Performing {NUM_WARMUP_RUNS} warm-up runs on {os.path.basename(first_image)}...")
        for _ in range(NUM_WARMUP_RUNS):
            _ = model.predict(first_image, device=device, verbose=False) # Keep verbose False
        if device == "cuda":
            torch.cuda.synchronize()
        print("  Warm-up complete.")

        # Timed runs on the first image
        print(f"  Performing {NUM_TIMED_RUNS} timed runs on {os.path.basename(first_image)}...")
        for _ in range(NUM_TIMED_RUNS):
            # No need for external timing, use the result's speed info
            results = model.predict(first_image, device=device, verbose=False) # Keep verbose False
            if device == "cuda":
                # Synchronization might still be needed if predict itself doesn't guarantee it
                # before returning the speed results, although it likely does.
                # Keeping it for safety, especially for accurate GPU timing.
                torch.cuda.synchronize()
            # Extract inference time (already in ms)
            if results and len(results) > 0 and hasattr(results[0], 'speed') and 'inference' in results[0].speed:
                 inference_latencies_ms.append(results[0].speed['inference'])
            else:
                print("  Warning: Could not extract inference time from results.")
                # Optionally, fall back to perf_counter or skip this run
                # For now, we just skip appending if data is missing

        if not inference_latencies_ms:
             print("  Error: No valid inference times recorded.")
             return None

        mean_latency_ms = statistics.mean(inference_latencies_ms)
        median_latency_ms = statistics.median(inference_latencies_ms)

        print(f"  Mean Inference Latency over {len(inference_latencies_ms)} runs: {mean_latency_ms:.3f} ms")
        print(f"  Median Inference Latency over {len(inference_latencies_ms)} runs: {median_latency_ms:.3f} ms")
        # Return a dictionary containing both values
        return {"mean": mean_latency_ms, "median": median_latency_ms}

    except Exception as e:
        print(f"  Error during benchmarking for {model_path}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find images
    image_search_path = os.path.join(IMAGE_DIR, IMAGE_PATTERN)
    image_paths = glob.glob(image_search_path)

    if not image_paths:
        print(f"Error: No images found matching '{IMAGE_PATTERN}' in directory '{IMAGE_DIR}'")
        exit()
    else:
        print(f"Found {len(image_paths)} images for benchmarking in '{IMAGE_DIR}'.")
        # Using the first image for benchmarking runs
        print(f"Using image: {os.path.basename(image_paths[0])}")
        print()

    results = {}

    # Benchmark each model configuration
    for model_name, paths in MODEL_CONFIG.items():
        print(f"--- Benchmarking {model_name} ---")
        results[model_name] = {}

        original_path = os.path.join(MODEL_DIR, paths["original"])
        smashed_path = os.path.join(MODEL_DIR, paths["smashed"])

        # Benchmark original
        original_metrics = benchmark_model(original_path, image_paths, device)
        results[model_name]["original_latency"] = original_metrics # Store the dict

        # Benchmark smashed
        smashed_metrics = benchmark_model(smashed_path, image_paths, device)
        results[model_name]["smashed_latency"] = smashed_metrics # Store the dict
        print("-" * (len(f"--- Benchmarking {model_name} ---")))
        print("") # Newline for separation

    # --- Print Summary ---
    print("\n--- Comparison Summary ---")
    print(f"Device: {device}")
    print(f"Image used for timing: {os.path.basename(image_paths[0]) if image_paths else 'N/A'}") # Renamed for clarity
    print(f"Warmup runs: {NUM_WARMUP_RUNS}")
    print(f"Timed runs: {NUM_TIMED_RUNS}")
    print("-" * 26)

    for model_name, data in results.items():
        print(f"\nModel: {model_name}")
        original_metrics = data.get("original_latency")
        smashed_metrics = data.get("smashed_latency")

        print("  Original:")
        if original_metrics:
            print(f"    Mean Latency:   {original_metrics['mean']:.3f} ms")
            print(f"    Median Latency: {original_metrics['median']:.3f} ms")
        else:
            print("    Latency: Error / Not Measured")

        print("  Smashed:")
        if smashed_metrics:
            print(f"    Mean Latency:   {smashed_metrics['mean']:.3f} ms")
            print(f"    Median Latency: {smashed_metrics['median']:.3f} ms")
        else:
            print("    Latency: Error / Not Measured")

        # Calculate speedup based on mean latency
        if original_metrics and smashed_metrics and original_metrics['mean'] > 0:
            mean_speedup = original_metrics['mean'] / smashed_metrics['mean']
            median_speedup = original_metrics['median'] / smashed_metrics['median']
            print(f"  Speedup (Mean):   {mean_speedup:.2f}x")
            print(f"  Speedup (Median): {median_speedup:.2f}x")
        elif original_metrics and smashed_metrics:
             print("  Speedup: N/A (Original latency zero or error)")
        else:
            print("  Speedup: N/A (Error in measurement)")

    print("\n--- End Summary ---")
    print("Script finished.") 