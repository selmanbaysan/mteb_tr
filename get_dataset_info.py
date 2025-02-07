from datasets import get_dataset_config_info
from datasets import load_dataset
import argparse

def get_revision(dataset_path: str) -> str:
    """Get the latest revision number for a Hugging Face dataset.
    
    Args:
        dataset_path: Path to dataset on Hugging Face Hub (e.g. 'davidstap/sranantongo')
        
    Returns:
        str: Latest revision number for the dataset
    """
    try:
        info = get_dataset_config_info(dataset_path)
        return info
    except Exception as e:
        print(f"Error getting revision for {dataset_path}: {str(e)}")
        return None


def read_dataset(dataset_path: str):
    dataset = load_dataset(dataset_path)
    print(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get dataset revision number')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset on HF Hub')
    args = parser.parse_args()

    revision = get_revision(args.dataset_path)
    if revision:
        print(f"Latest revision for {args.dataset_path}: {revision}")

    read_dataset(args.dataset_path)
