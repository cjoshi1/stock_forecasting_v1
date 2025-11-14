"""
Kaggle data downloader for Rossmann Store Sales competition.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional


def check_data_exists(data_dir: str = 'rossman_forecasting/data/raw') -> bool:
    """
    Check if Rossmann data files already exist.

    Args:
        data_dir: Directory to check for data files

    Returns:
        True if all required files exist, False otherwise
    """
    required_files = ['train.csv', 'test.csv', 'store.csv']
    data_path = Path(data_dir)

    if not data_path.exists():
        return False

    for file in required_files:
        if not (data_path / file).exists():
            return False

    print(f"✅ All required files found in {data_dir}")
    return True


def download_rossmann_data(
    competition: str = 'rossmann-store-sales',
    data_dir: str = 'rossman_forecasting/data/raw',
    force: bool = False
) -> bool:
    """
    Download Rossmann Store Sales data from Kaggle.

    Args:
        competition: Kaggle competition name
        data_dir: Directory to save downloaded files
        force: If True, download even if files exist

    Returns:
        True if download successful, False otherwise

    Raises:
        RuntimeError: If Kaggle API is not configured
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    if not force and check_data_exists(data_dir):
        print(f"📁 Data already exists in {data_dir}")
        print("   Use force=True or --force flag to re-download")
        return True

    # Check if Kaggle API is configured
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        raise RuntimeError(
            "Kaggle API not configured. Please follow these steps:\n"
            "1. Go to https://www.kaggle.com/account\n"
            "2. Click 'Create New API Token' to download kaggle.json\n"
            "3. Place it at ~/.kaggle/kaggle.json\n"
            "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        )

    print(f"📥 Downloading Rossmann data from Kaggle...")
    print(f"   Competition: {competition}")
    print(f"   Destination: {data_dir}")

    try:
        # Download using Kaggle API
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', competition, '-p', str(data_path)],
            capture_output=True,
            text=True,
            check=True
        )

        print("✅ Download completed")

        # Check if files were downloaded as zip
        zip_file = data_path / f'{competition}.zip'
        if zip_file.exists():
            print(f"📦 Extracting {zip_file.name}...")
            subprocess.run(
                ['unzip', '-o', str(zip_file), '-d', str(data_path)],
                capture_output=True,
                check=True
            )
            zip_file.unlink()  # Remove zip after extraction
            print("✅ Extraction completed")

        # Verify all files exist
        if check_data_exists(data_dir):
            print("✅ All required files downloaded successfully")
            return True
        else:
            print("⚠️  Some files may be missing. Please check the data directory.")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e.stderr}")
        print("\nPossible issues:")
        print("1. Kaggle API not installed: pip install kaggle")
        print("2. Competition rules not accepted: Visit https://www.kaggle.com/c/rossmann-store-sales/rules")
        print("3. Network issues: Check your internet connection")
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        return False


def list_data_files(data_dir: str = 'rossman_forecasting/data/raw') -> None:
    """
    List all files in the data directory with sizes.

    Args:
        data_dir: Directory to list files from
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    print(f"\n📁 Files in {data_dir}:")
    print("-" * 60)

    files = sorted(data_path.glob('*.csv'))
    if not files:
        print("   No CSV files found")
        return

    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {file.name:<30} {size_mb:>8.2f} MB")

    print("-" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download Rossmann Store Sales data from Kaggle')
    parser.add_argument('--force', action='store_true', help='Force re-download even if files exist')
    parser.add_argument('--data_dir', type=str, default='rossman_forecasting/data/raw',
                       help='Directory to save data files')
    parser.add_argument('--list', action='store_true', help='List existing data files')

    args = parser.parse_args()

    if args.list:
        list_data_files(args.data_dir)
    else:
        success = download_rossmann_data(data_dir=args.data_dir, force=args.force)
        if success:
            list_data_files(args.data_dir)
