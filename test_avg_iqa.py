import pyiqa
import torch
import os
import argparse
from pathlib import Path



def main():
    # EDIT via CLI: restored PNG folder, requested metrics and compute device.
    # These are no-reference scores; do not include GT or noisy inputs in the folder.
    parser = argparse.ArgumentParser(description='Average optional no-reference IQA scores on restored images.')
    parser.add_argument('--image_folder', type=str, default='results/sidd_ANT_sidd', help='Restored image directory, searched recursively.')
    parser.add_argument('--metrics', type=str, nargs='+', default=['maniqa', 'clipiqa', 'musiq'],
                        help='list of pyiqa metric names to evaluate')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, cuda, or cuda:0')
    args = parser.parse_args()
    
    # List available models
    available = pyiqa.list_models()
    print("Available metrics:", available)


    # Device configuration
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Prepare metrics (skip unavailable)
    metrics = {}
    for name in args.metrics:
        if name not in available:
            print(f"Warning: metric '{name}' not available, skipping.")
            continue
        try:
            metrics[name] = pyiqa.create_metric(name, device=device)
        except Exception as e:
            print(f"Failed to create metric '{name}': {e}")

    if len(metrics) == 0:
        print("No valid metrics to evaluate. Exiting.")
        return

    # Collect image files
    image_folder = args.image_folder
    if not os.path.isdir(image_folder):
        parser.error('Image directory not found: ' + image_folder)
    image_files = [str(p.relative_to(image_folder)) for p in Path(image_folder).rglob('*')
                   if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    image_files.sort()
    
    if len(image_files) == 0:
        print("No images found in", image_folder)
        return

    # Accumulate scores
    sums = {name: 0.0 for name in metrics}
    counts = {name: 0 for name in metrics}

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Evaluating {image_file} ...")
        for name, metric in metrics.items():
            try:
                score = metric(image_path)
                # metric may return tensor or float
                val = float(score.item() if hasattr(score, 'item') else score)
                sums[name] += val
                counts[name] += 1
                print(f"  {name}: {val:.6f}")
            except Exception as e:
                print(f"  Error evaluating {name} on {image_file}: {e}")

    # Print averages
    print("\nAverage results:")
    for name in metrics:
        if counts[name] > 0:
            avg = sums[name] / counts[name]
            lower_better = getattr(metrics[name], 'lower_better', None)
            lb_text = " (lower better)" if lower_better else ""
            print(f"  {name}: {avg:.6f}{lb_text} (scored {counts[name]}/{len(image_files)} images)")
        else:
            print(f"  {name}: no valid scores")

if __name__ == "__main__":
    main()
