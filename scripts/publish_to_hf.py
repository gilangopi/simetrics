"""
publish_to_hf.py , push the frozen SiMetrics dataset to a (PRIVATE) Hugging Face dataset repo.

ONE-TIME SETUP (do this yourself; needs YOUR Hugging Face account):
    pip install huggingface_hub
    huggingface-cli login            # paste a WRITE token from https://huggingface.co/settings/tokens

THEN, from the repo root (simetrics/):
    python scripts/publish_to_hf.py <your-username>/simetrics            # creates it PRIVATE
    python scripts/publish_to_hf.py <your-username>/simetrics --public   # only when you're ready

What it does:
  * creates the dataset repo (private by default, exist_ok=True),
  * uploads the dataset card (scripts/hf_dataset_card.md) as the repo README.md,
  * uploads the whole data/simetrics_benchmark/ folder (34 .npz + datasheet + validation + summary),
  * tags the commit so you have a citable, frozen version.

It NEVER force-pushes and prints the repo URL at the end. Flip to public later with --public
(or in the repo Settings on huggingface.co).
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # simetrics/ repo root
DATA_DIR = os.path.join(ROOT, "data", "simetrics_benchmark")
CARD = os.path.join(HERE, "hf_dataset_card.md")
VERSION_TAG = "v0.3.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="target dataset repo, e.g. your-username/simetrics")
    ap.add_argument("--public", action="store_true", help="create/keep the repo PUBLIC (default: private)")
    ap.add_argument("--no-tag", action="store_true", help="skip creating the version tag")
    args = ap.parse_args()

    if not os.path.isdir(DATA_DIR):
        sys.exit(f"dataset not found at {DATA_DIR}\n  -> generate it first: python scripts/make_benchmark.py")
    n_npz = sum(f.endswith(".npz") for _, _, fs in os.walk(DATA_DIR) for f in fs)
    if n_npz < 34:
        print(f"WARNING: expected 34 .npz, found {n_npz} (regenerate if this looks wrong).")

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    try:
        who = api.whoami().get("name", "?")
    except Exception:
        sys.exit("Not logged in. Run:  huggingface-cli login   (use a WRITE token)")
    print(f"logged in as: {who}")

    private = not args.public
    print(f"creating dataset repo '{args.repo_id}' (private={private}, exist_ok=True) ...")
    create_repo(args.repo_id, repo_type="dataset", private=private, exist_ok=True)

    print("uploading dataset card -> README.md ...")
    api.upload_file(path_or_fileobj=CARD, path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="dataset",
                    commit_message="dataset card")

    print(f"uploading {n_npz} arrays + docs from {DATA_DIR} ...")
    api.upload_folder(folder_path=DATA_DIR, path_in_repo="", repo_id=args.repo_id,
                      repo_type="dataset", ignore_patterns=["README.md", "*.pyc", "__pycache__/*"],
                      commit_message=f"frozen dataset {VERSION_TAG}")

    if not args.no_tag:
        try:
            api.create_tag(args.repo_id, tag=VERSION_TAG, repo_type="dataset")
            print(f"tagged {VERSION_TAG}")
        except Exception as e:
            print(f"(tag skipped: {e})")

    url = f"https://huggingface.co/datasets/{args.repo_id}"
    print(f"\nDONE -> {url}")
    print("  PRIVATE" if private else "  PUBLIC")
    print("  When ready for the public release, either re-run with --public,")
    print("  or flip visibility in the repo Settings on huggingface.co.")


if __name__ == "__main__":
    main()
