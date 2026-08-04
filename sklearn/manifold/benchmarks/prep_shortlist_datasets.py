"""
Wire up the Experiment-K shortlist from the bulk collections into the standard
datasets/<name>/X.npy layout that load_local_dataset expects.

The collections (hf_datasets_p1/p2, pmlb_datasets, scanpy_datasets,
sklearn_datasets) are flat directories of single .npy feature matrices. Rather
than duplicate gigabytes, this creates datasets/<name>/ with X.npy as a
RELATIVE SYMLINK to the source file (robust to the datasets/ dir moving; only
breaks if the source collection is deleted). load_local_dataset standardizes on
load, so no pre-normalisation is needed and the stored float64 is upcast anyway.

Shortlist (regime = 2D-embeddability on standardized data; verified on load):
  Responsive, spanning D  -> tests the D-vs-speed axis (H-K3):
    feynman_iii_15_12  D=3     (extreme low-D -> biggest speedup)
    california_housing D=8     (low-D, real)
    patchcamelyon      D=512   (mid-D image)
    cifar10_raw        D=3072  (very high-D -> smallest speedup)
  Hard-middle, large N    -> should NOT reach constant-k (H-K4 falsification):
    ag_news            D=384   (text)
    tiny_imagenet      D=512   (image)
    mnist_digits       D=784   (digit MNIST; less 2D-embeddable than fashion)

Run once (from this benchmarks/ directory):
    python prep_shortlist_datasets.py
"""

from pathlib import Path
import os

DATASETS_DIR = Path(__file__).parent / "datasets"

# short_name -> (collection_dir, source_filename)
SHORTLIST = {
    "feynman_iii_15_12": ("pmlb_datasets", "pmlb_feynman_III_15_12.npy"),
    "california_housing": ("sklearn_datasets", "sklearn_california_housing.npy"),
    "ag_news": ("hf_datasets_p1", "hf_text_fancyzhx__ag_news.npy"),
    "mnist_digits": ("pmlb_datasets", "pmlb_mnist.npy"),
    "patchcamelyon": ("hf_datasets_p2", "hf_image_1aurent__PatchCamelyon.npy"),
    "tiny_imagenet": ("hf_datasets_p2", "hf_image_zh-plus__tiny-imagenet.npy"),
    "cifar10_raw": ("hf_datasets_p1", "hf_image_uoft-cs__cifar10_raw.npy"),
}


def main():
    for name, (coll, fname) in SHORTLIST.items():
        src = DATASETS_DIR / coll / fname
        if not src.exists():
            print(f"  !! SOURCE MISSING for {name}: {src}")
            continue
        out_dir = DATASETS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        link = out_dir / "X.npy"
        if link.exists() or link.is_symlink():
            link.unlink()
        # relative target: ../<coll>/<fname>  (from datasets/<name>/)
        rel = os.path.relpath(src, out_dir)
        link.symlink_to(rel)
        print(f"  {name:<20} -> {rel}")
    print("\nDone. Verify with load_local_dataset('<name>').")


if __name__ == "__main__":
    main()
