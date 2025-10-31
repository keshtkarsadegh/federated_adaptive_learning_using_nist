import os
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
from src.federated_adaptive_learning_nist.nist_logger import NistLogger


def download_and_extract_shell(url, extract_to="nist_data"):
    """
    Downloads and extracts a ZIP file using shell commands (wget + unzip)
    with a progress bar, skipping download if the file already exists.
    """
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split("/")[-1]
    zip_path = os.path.join(extract_to, filename)

    try:
        # === DOWNLOAD (skip if exists) ===
        if not os.path.exists(zip_path):
            NistLogger.info(f"Downloading {url} -> {zip_path}")
            subprocess.run(
                ["wget", "-O", zip_path, "--progress=bar:force:noscroll", url],
                check=True,
            )
        else:
            NistLogger.info(f"ZIP already exists, skipping download: {zip_path}")

        # === COUNT FILES ===
        total_files = int(
            subprocess.check_output(
                f'unzip -Z1 "{zip_path}" | wc -l', shell=True
            ).strip()
        )

        NistLogger.info(f"Extracting {zip_path} to {extract_to} ({total_files} files)...")
        start_time = time.time()

        # === EXTRACT WITH PROGRESS ===
        proc = subprocess.Popen(
            ["unzip", "-o", zip_path, "-d", extract_to],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
            for line in proc.stdout:
                if line.startswith("  inflating:") or line.startswith(" extracting:"):
                    pbar.update(1)
        proc.wait()

        elapsed = time.time() - start_time
        NistLogger.info(f"Extraction complete in {elapsed:.1f}s → {extract_to}")

        # === CLEANUP ===
        os.remove(zip_path)
        NistLogger.info(f"Deleted archive: {zip_path}")

    except subprocess.CalledProcessError as e:
        NistLogger.error(f"Shell command failed: {e}")
    except Exception as e:
        NistLogger.error(f"Unexpected error: {e}")


import os
import requests
from pathlib import Path

def download_hash_by_class(url: str, target_dir: str):
    """
    Downloads a file from the given URL into the target directory.
    Shows progress and skips download if file already exists.
    """
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.basename(url)
    target_path = Path(target_dir) / filename

    if target_path.exists():
        print(f"File already exists: {target_path}")
        return str(target_path)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB

    downloaded = 0
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = downloaded / total_size * 100
                print(f"\rDownloading {filename}: {percent:.1f}% ({downloaded/1024/1024:.2f} MB)", end="")
    print(f"\nDownload complete: {target_path}")

    return str(target_path)


if __name__ == "__main__":
    url = "https://s3.amazonaws.com/nist-srd/SD19/by_class_md5.log"
    target_dir = "./nist_data"

    download_file(url, target_dir)



if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    PROJECT_ROOT = script_dir.parent.parent
    target_dir = str(PROJECT_ROOT / "data")

    urls = [
        "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip",
        "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
    ]

    for u in urls:
        download_and_extract_shell(u, extract_to=target_dir)
