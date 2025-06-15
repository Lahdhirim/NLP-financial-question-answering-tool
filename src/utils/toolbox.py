import pandas as pd
from colorama import Fore, Style

def load_csv_data(data_path: str) -> pd.DataFrame:
        try :
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            raise print(Fore.RED + f"Could not find the CSV file at {data_path}. Please check the path and try again." + Style.RESET_ALL)
        return data