import argparse

# Import config loaders and pipeline classes for each task
from src.config_loaders.preprocessing_config_loader import preprocessing_config_loader
from src.preprocessing_pipeline import PreprocessingPipeline


if __name__ == "__main__":
    """
    Main script for running the Financial QA Tool in different operational modes.

    This script provides a command-line interface to perform various tasks related to data preprocessing, model training, evaluation, and inference.
    Depending on the selected mode, it loads the appropriate configuration and executes the corresponding pipeline.
    """

    # Parse command-line argument to determine which mode to run
    parser = argparse.ArgumentParser(description="Financial QA Tool")
    parser.add_argument("mode", choices=["process_data", "train", "evaluate", "inference"],
                        default="process_data", nargs="?", help="Choose mode: process_data, train, evaluate or inference")
    args = parser.parse_args()

    # Launch the appropriate pipeline based on the selected mode
    if args.mode == "process_data":
        # Load processing config and run data preprocessing pipeline
        processing_config = preprocessing_config_loader(config_path="config/preprocessing_config.json")
        processing_pipeline = PreprocessingPipeline(config=processing_config)
        processing_pipeline.run()
    
    else:
        print("Invalid mode. Please choose 'process_data', 'train', 'evaluate', or 'inference'.")