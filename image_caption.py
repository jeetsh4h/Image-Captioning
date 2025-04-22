import argparse
import time
import torch
import os
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
from datetime import timedelta, datetime
from transformers.models.bert import BertTokenizer
import glob
import matplotlib.font_manager as fm
import textwrap
import math
import numpy as np  # Needed for generate_grid

# Assuming utils, models, evaluation are available
from utils import transform
from models import ImageCaptionModel
from evaluation import generate_caption

# --- Grid Generation Logic (Adapted from the OpenCV script) ---


def updateGrid(grid, x, y, counter, n):
    """Helper for spiral grid generation."""
    if 0 <= x < n and 0 <= y < n:
        # Check if already filled (important for odd n starting center)
        if grid[x][y] == -1:  # Use -1 for unfilled initially
            grid[x][y] = counter
            return counter + 1
    return counter  # Return same counter if outside bounds or already filled


def generate_grid(
    n, grid_type, invert_chirality, grid_flip_up_down, grid_flip_left_right
):
    """Generates a 2D grid layout mapping cell position to sequential image index."""
    grid = np.full((n, n), -1, dtype=int)  # Initialize with -1

    if grid_type == "spiral":
        right = lambda x, y: (x, y + 1)
        left = lambda x, y: (x, y - 1)
        up = lambda x, y: (x - 1, y)
        down = lambda x, y: (x + 1, y)
        direction_mapping = {0: right, 1: up, 2: left, 3: down}
        # Adjust direction based on chirality (0: RULD, 1: LDRU if inverted)
        # Original logic seemed complex, simplifying: RULD normally, LDRU if inverted?
        # Let's stick to the original's intent if possible, but note it might need adjustment
        # find_direction = lambda i, inv: (i + 2 * inv) % 4 if i % 2 == 0 else i % 4 # Original logic
        # Simpler RULD/LDRU based on chirality
        if invert_chirality:
            # LDRU equivalent directions after map lookup
            dir_order = [
                direction_mapping[2],
                direction_mapping[3],
                direction_mapping[1],
                direction_mapping[0],
            ]
        else:
            # RULD order
            dir_order = [
                direction_mapping[0],
                direction_mapping[1],
                direction_mapping[2],
                direction_mapping[3],
            ]

        x = n // 2
        y = n // 2  # Start at true center for spiral
        counter = 0
        grid[x][y] = counter  # Place the first image (index 0) at the center
        counter += 1

        # Start moving right (or left if inverted)
        move_dir_idx = 0 if not invert_chirality else 2
        steps = 1

        while counter < n * n:
            for _ in range(2):  # Change direction twice per step increase
                move_func = direction_mapping[move_dir_idx % 4]
                for _ in range(steps):
                    x, y = move_func(x, y)
                    if 0 <= x < n and 0 <= y < n:
                        if grid[x][y] == -1:
                            grid[x][y] = counter
                            counter += 1
                    if counter >= n * n:
                        break  # Exit early if grid filled
                if counter >= n * n:
                    break
                # Turn counter-clockwise (or clockwise if inverted)
                move_dir_idx += 1 if not invert_chirality else -1

            steps += 1

    elif grid_type == "normal" or grid_type == "snake":
        counter = 0
        temp_grid = np.arange(n * n).reshape((n, n))

        if grid_type == "snake":
            temp_grid[1::2, :] = temp_grid[1::2, ::-1]  # Flip odd rows (1, 3, 5...)

        # Apply transpose for chirality *before* assigning to the final grid
        if invert_chirality:
            temp_grid = temp_grid.T  # Transpose

        grid = temp_grid  # Assign the calculated indices

    else:  # Default to normal if type is unknown
        print(f"Warning: Unknown grid_type '{grid_type}'. Defaulting to 'normal'.")
        grid = np.arange(n * n).reshape((n, n))

    # Apply Flips after layout generation
    if grid_flip_up_down:
        grid = np.flipud(grid)
    if grid_flip_left_right:
        grid = np.fliplr(grid)

    # Final check: any -1 left implies calculation error, replace with placeholder index?
    # For now, assume it fills correctly or errors out if counter doesn't reach n*n

    return grid


# --- End Grid Generation Logic ---


def get_next_file_index(output_dir, prefix="caption_visualization"):
    """Find the next available index for saving visualization files"""
    existing_files = glob.glob(os.path.join(output_dir, f"{prefix}_*.png"))
    indices = [0]
    for file in existing_files:
        try:
            indices.append(int(os.path.basename(file).split("_")[-1].split(".")[0]))
        except (ValueError, IndexError):
            continue
    return max(indices) + 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate Image Caption Visualizations in a Grid"
    )
    # --- Model parameters ---
    parser.add_argument("--embedding_dim", "-ed", type=int, default=512)
    parser.add_argument("--tokenizer", "-t", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_len", "-msl", type=int, default=128)
    parser.add_argument("--encoder_layers", "-ad", type=int, default=6)
    parser.add_argument("--decoder_layers", "-nl", type=int, default=12)
    parser.add_argument("--num_heads", "-nh", type=int, default=8)
    parser.add_argument("--dropout", "-dr", type=float, default=0.1)
    parser.add_argument(
        "--model_path", "-md", type=str, required=True, help="Path to saved model"
    )

    # --- Data parameters ---
    parser.add_argument(
        "--image_dir",
        "-id",
        type=str,
        required=True,
        help="Path to image directory (e.g., ./coco/)",
    )
    parser.add_argument(
        "--karpathy_json_path",
        "-kap",
        type=str,
        required=True,
        help="Path to karpathy json file",
    )
    parser.add_argument(
        "--phase",
        "-p",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset phase",
    )

    # --- Generation parameters ---
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--beam_size", "-b", type=int, default=3)

    # --- Output & Grid parameters ---
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./visualizations/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_images",
        "-n",
        type=int,
        default=9,
        help="Max number of images to use (forms grid)",
    )
    parser.add_argument(
        "--caption_width",
        "-cw",
        type=int,
        default=42,
        help="Max width for wrapping captions",
    )
    parser.add_argument(
        "--no_individual",
        "-ni",
        action="store_true",
        help="Do not save individual images",
    )

    # --- Grid Layout parameters (from OpenCV script) ---
    parser.add_argument(
        "--grid_type",
        type=str,
        default="normal",
        choices=["normal", "snake", "spiral"],
        help="Grid filling pattern",
    )
    parser.add_argument(
        "--invert_chirality",
        action="store_true",
        help="Transpose grid layout (spiral/snake)",
    )
    parser.add_argument(
        "--grid_flip_up_down", action="store_true", help="Flip grid vertically"
    )
    parser.add_argument(
        "--grid_flip_left_right", action="store_true", help="Flip grid horizontally"
    )
    # --grid-size from OpenCV script is handled by --num_images here
    # --force-odd-size is specific to spiral needing odd N, not directly implemented here yet

    args = parser.parse_args()

    # --- Setup ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model_configs = {
        k: getattr(args, k)
        for k in [
            "embedding_dim",
            "max_seq_len",
            "encoder_layers",
            "decoder_layers",
            "num_heads",
            "dropout",
        ]
    }
    model_configs["vocab_size"] = tokenizer.vocab_size

    start_time = time.time()
    model = ImageCaptionModel(**model_configs)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(
        f"Model loaded on {device} in {timedelta(seconds=int(time.time() - start_time))}"
    )

    print(f"Loading dataset from {args.karpathy_json_path}")
    karpathy_data = json.load(open(args.karpathy_json_path, "r"))
    phase_filters = {"train", "restval"} if args.phase == "train" else {args.phase}
    all_phase_images = [
        img for img in karpathy_data["images"] if img["split"] in phase_filters
    ]
    print(f"Found {len(all_phase_images)} images in {args.phase} set")
    if not all_phase_images:
        print(f"Error: No images found.")
        return

    # Select images up to num_images
    selected_images = random.sample(
        all_phase_images, min(args.num_images, len(all_phase_images))
    )
    num_images_to_plot = len(selected_images)
    if num_images_to_plot == 0:
        print("No images selected.")
        return
    print(f"Selected {num_images_to_plot} random images for visualization")

    # --- Calculate Grid Dimensions ---
    # Make it as square as possible based on num_images_to_plot
    ncols = math.ceil(math.sqrt(num_images_to_plot))
    nrows = math.ceil(num_images_to_plot / ncols)
    grid_size_n = max(
        nrows, ncols
    )  # Use the larger dimension for generate_grid if not square
    total_cells = nrows * ncols

    print(f"Creating a {nrows}x{ncols} grid layout ({total_cells} cells total).")
    if args.grid_type == "spiral" and grid_size_n % 2 == 0:
        print(
            f"Warning: Spiral grid works best with odd dimensions (N={grid_size_n}). Result might be off-center."
        )

    # --- Generate Grid Layout ---
    # The generate_grid function returns indices from 0 to N*N-1
    grid_layout = generate_grid(
        grid_size_n,  # Use the calculated dimension
        args.grid_type,
        args.invert_chirality,
        args.grid_flip_up_down,
        args.grid_flip_left_right,
    )

    # --- Font Setup ---
    monospace_fonts = ["DejaVu Sans Mono", "Courier New", "Consolas", "Inconsolata"]
    monospace_font = "monospace"
    for font in monospace_fonts:
        try:
            fm.findfont(font, fallback_to_default=False)
            monospace_font = font
            print(f"Using font: {monospace_font}")
            break
        except ValueError:
            continue
    if monospace_font == "monospace":
        print("Using default 'monospace'.")

    # --- Create Figure and Subplots ---
    # Estimate figure size needed based on number of rows/cols and desired spacing
    # These factors might need tuning
    base_img_size_inch = 4.0  # Assumed size for image display area
    caption_space_inch = 1.5  # Estimated space needed below image
    fig_width = ncols * (base_img_size_inch + 0.5)  # Add some horizontal padding
    fig_height = nrows * (base_img_size_inch + caption_space_inch)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), facecolor="white", squeeze=False
    )
    # Adjust spacing AFTER creating subplots
    # hspace is height spacing (fraction of axes height), wspace is width
    # Increase hspace significantly to make room for captions below
    fig.subplots_adjust(
        hspace=0.6, wspace=0.15, top=0.95, bottom=0.05, left=0.05, right=0.95
    )

    # --- Plot Images and Captions according to Grid Layout ---
    plotted_count = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")  # Turn off axes for all cells initially

            # Map the cell (r, c) to the sequential index from grid_layout
            # Need to handle potential size mismatch if grid_layout is larger (e.g., from odd spiral)
            if r < grid_size_n and c < grid_size_n:
                image_seq_index = grid_layout[r, c]
            else:
                image_seq_index = -1  # Index is outside the generated layout

            # Check if the index corresponds to a valid selected image
            if 0 <= image_seq_index < num_images_to_plot:
                img_data = selected_images[image_seq_index]
                plotted_count += 1

                img_path = os.path.join(
                    args.image_dir, img_data["filepath"], img_data["filename"]
                )
                print(
                    f"  Plotting cell ({r},{c}) <- Image index {image_seq_index}: {img_path}"
                )

                if not os.path.exists(img_path):
                    print(f"    Warn: Not found: {img_path}")
                    ax.text(
                        0.5,
                        0.5,
                        "Not found",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="red",
                        transform=ax.transAxes,
                    )
                    continue

                try:
                    true_caption = [sent["raw"] for sent in img_data["sentences"]][0]
                    predicted_caption = generate_caption(
                        model=model,
                        image_path=img_path,
                        transform_fn=transform,
                        tokenizer=tokenizer,
                        max_seq_len=args.max_seq_len,
                        beam_size=args.beam_size,
                        device=device,
                        print_process=False,
                    )

                    wrapped_true = "\n".join(
                        textwrap.wrap(true_caption, width=args.caption_width)
                    )
                    wrapped_pred = "\n".join(
                        textwrap.wrap(predicted_caption, width=args.caption_width)
                    )

                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)

                    # --- Render Captions Below ---
                    fontsize = 9
                    label_fontsize = 9
                    fontfamily = monospace_font
                    x_pos = 0.05  # Keep text away from left edge
                    ha = "left"
                    va = "top"
                    # Adjust y-positions relative to axes (0,0 is bottom-left, 1,1 is top-right)
                    # Need negative y to go below the axes
                    y_true_label = -0.07  # Start below axes
                    line_sep = 0.045  # Estimated height per line
                    block_sep = 0.08  # Gap between true/pred blocks

                    y_true_caption = y_true_label - line_sep * 0.9
                    true_caption_height = (wrapped_true.count("\n") + 1) * line_sep
                    y_pred_label = y_true_caption - true_caption_height - block_sep
                    y_pred_caption = y_pred_label - line_sep * 0.9

                    ax.text(
                        x_pos,
                        y_true_label,
                        "True:",
                        fontfamily=fontfamily,
                        fontsize=label_fontsize,
                        fontweight="bold",
                        ha=ha,
                        va=va,
                        wrap=False,
                        transform=ax.transAxes,
                    )
                    ax.text(
                        x_pos,
                        y_true_caption,
                        wrapped_true,
                        fontfamily=fontfamily,
                        fontsize=fontsize,
                        fontweight="normal",
                        ha=ha,
                        va=va,
                        wrap=False,
                        transform=ax.transAxes,
                    )
                    ax.text(
                        x_pos,
                        y_pred_label,
                        "Predicted:",
                        fontfamily=fontfamily,
                        fontsize=label_fontsize,
                        fontweight="bold",
                        ha=ha,
                        va=va,
                        wrap=False,
                        transform=ax.transAxes,
                    )
                    ax.text(
                        x_pos,
                        y_pred_caption,
                        wrapped_pred,
                        fontfamily=fontfamily,
                        fontsize=fontsize,
                        fontweight="normal",
                        ha=ha,
                        va=va,
                        wrap=False,
                        transform=ax.transAxes,
                    )

                    # --- Individual Save (Optional) ---
                    # ... (individual saving code can remain similar, using img_data, wrapped_true, wrapped_pred) ...
                    if not args.no_individual:
                        # ... (omitted for brevity, but copy/adapt from previous version) ...
                        pass  # Placeholder

                except FileNotFoundError:
                    print(f"    Error: Not found: {img_path}")
                    ax.text(
                        0.5,
                        0.5,
                        "Error loading",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="red",
                        transform=ax.transAxes,
                    )
                except Exception as e:
                    print(f"    Error processing {img_path}: {e}")
                    ax.text(
                        0.5,
                        0.5,
                        "Proc. Error",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="red",
                        transform=ax.transAxes,
                    )

            # else: # Cell corresponds to an index >= num_images_to_plot or -1
            # ax.axis('off') # Already off by default above
            # Optionally add a placeholder to truly empty cells
            # ax.text(0.5, 0.5, "-", ha="center", va="center", fontsize=20, color="lightgray", transform=ax.transAxes)
            # pass

    print(f"Plotted {plotted_count} images.")

    # --- Save Final Grid ---
    file_index = get_next_file_index(
        args.output_dir, prefix=f"caption_grid_{args.grid_type}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = f"caption_grid_{args.grid_type}_{args.phase}_{nrows}x{ncols}_{timestamp}_{file_index}"
    output_file_path = os.path.join(
        args.output_dir, output_filename_base + ".png"
    )  # Default to PNG

    # No need for manual grid lines with subplots, spacing controlled by subplots_adjust
    # Optional: Add overall title
    # fig.suptitle(f"Image Captions ({args.grid_type} layout)", fontsize=16)
    # fig.subplots_adjust(top=0.92) # Adjust top margin if suptitle is used

    plt.savefig(output_file_path, dpi=150, pad_inches=0.2)  # Use bbox_inches and pad
    print(f"\nGrid visualization saved to {output_file_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
