import os
import random
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split each category‐folder under data_root into train/test lists."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config (must contain 'seed', 'dataset.name', 'dataset.data_root', and 'dataset.categories')."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Fraction of subdirectories to include in <category>_train.txt (default: 0.8)."
    )
    return parser.parse_args()

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def split_subdirs(root_dir, seed, ratio, dataset_name, output_base):
    """
    - root_dir:        path that contains many subfolders (e.g. /…/data_root/02958343)
    - seed:            integer seed for shuffling
    - ratio:           fraction to put into train
    - dataset_name:    e.g. 'shapenet' or 'objaverse'
    - output_base:     base folder under which to write splits, e.g. 'datasets/splits'
    """
    if not os.path.isdir(root_dir):
        print(f"Warning: '{root_dir}' does not exist or is not a directory. Skipping.")
        return

    subdirs = [
        name
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    if not subdirs:
        print(f"No subdirectories found under '{root_dir}'. Skipping.")
        return

    # Shuffle with the given seed
    random.seed(seed)
    random.shuffle(subdirs)

    n_total = len(subdirs)
    n_train = int(ratio * n_total)
    train_list = subdirs[:n_train]
    test_list  = subdirs[n_train:]

    # Ensure the directory "datasets/splits/<dataset_name>/" exists
    split_dir = os.path.join(output_base, dataset_name)
    os.makedirs(split_dir, exist_ok=True)

    # Filenames under that directory
    # e.g. datasets/splits/shapenet/02958343_train.txt
    category_name = os.path.basename(root_dir.rstrip(os.sep))
    train_fname = os.path.join(split_dir, f"{category_name}_train.txt")
    test_fname  = os.path.join(split_dir, f"{category_name}_test.txt")

    with open(train_fname, "w") as f_train:
        for d in train_list:
            f_train.write(f"{d}\n")

    with open(test_fname, "w") as f_test:
        for d in test_list:
            f_test.write(f"{d}\n")

    print(f"[{category_name}] Found {n_total} subdirectories under '{root_dir}'.")
    print(f"  -> {len(train_list)} written to {train_fname}")
    print(f"  -> {len(test_list)} written to {test_fname}")

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # 1) Read seed, dataset.name, data_root, categories from YAML
    seed = cfg.get("seed", 42)

    dataset_cfg = cfg.get("dataset", {})
    dataset_name = dataset_cfg.get("name", None)
    data_root    = dataset_cfg.get("data_root", None)
    categories   = dataset_cfg.get("categories", [])

    if (dataset_name is None) or (data_root is None) or (not categories):
        print("Error: YAML must define 'dataset.name', 'dataset.data_root', and 'dataset.categories'.")
        return

    # Base folder for splits
    output_base = os.path.join("datasets", "splits")

    # For each category, split subfolders under data_root/<category>
    for cat in categories:
        root_dir = os.path.join(data_root, cat)
        split_subdirs(root_dir, seed, args.ratio, dataset_name, output_base)

if __name__ == "__main__":
    main()
