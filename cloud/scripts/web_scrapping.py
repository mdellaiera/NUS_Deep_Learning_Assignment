import os
import requests
from datetime import datetime, timedelta
import argparse


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the argument parser for command line arguments.
    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Process real clouds dataset.")
    parser.add_argument(
        "--start_date", type=str, required=True,
        help="Start date for downloading images (format: YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end_date", type=str, required=True,
        help="End date for downloading images (format: YYYY-MM-DD)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the images."
    )
    return parser


def download_images(start_date: str, end_date: str, output_dir: str) -> None:
    """
    Downloads satellite images from NEA from start_date to end_date (inclusive).
    
    Args:
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'
        output_dir (str): Directory to save images
    Returns:
        None
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Parse start and end dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Images are captured every 20 minutes
    delta = timedelta(minutes=20)

    # Loop through each timestamp
    current_dt = start_dt
    while current_dt <= end_dt:
        timestamp = current_dt

        # Build URL
        url = f"https://www.nea.gov.sg/docs/default-source/satelliteimage/BlueMarbleASEAN_{timestamp.strftime('%Y%m%d_%H%M')}.jpg"

        # Build output file path
        filename = f"BlueMarbleASEAN_{timestamp.strftime('%Y%m%d_%H%M')}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"Already exists: {filename}")
            current_dt += delta
            continue

        # Try to download
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {filename}")
            else:
                print(f"Failed ({response.status_code}): {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

        current_dt += delta


def main():
    """
    Main function to execute the script.
    """
    parser = build_argparser()
    args = parser.parse_args()
    download_images(args.start_date, args.end_date, args.output_dir)


if __name__ == "__main__":
    main()