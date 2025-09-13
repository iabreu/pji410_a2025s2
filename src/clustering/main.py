from src.helpers import extract
from src.clustering.load_csv import load_csv

def main(event):
    zip_filename = event.get("filename")
    password = event.get("password")
    extract(zip_filename, password)
    rows = load_csv()
    print(rows)


if __name__ == "__main__":
    event = {"filename": "fiscalizacao.csv.zip", "password": "XXXXX"}
    main(event)