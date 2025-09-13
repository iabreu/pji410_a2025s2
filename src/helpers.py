import zipfile
import os


def extract(zip_filename, password, extract_to="data"):
    zip_path = os.path.join("data", zip_filename)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file '{zip_path}' not found.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_names = zip_ref.namelist()
        zip_ref.extractall(path=extract_to, pwd=password.encode("utf-8"))
        return [f"{extract_to}/{x}" for x in file_names]
