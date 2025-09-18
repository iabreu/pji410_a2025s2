import zipfile
import os


class ZipExtractionError(Exception):
    """Erro ao extrair zip (senha incorreta ou arquivo corrompido)."""


def extract(zip_path: str, password: str, extract_to: str | None = None):
    """Extrai um arquivo zip protegido por senha.

    Args:
        zip_path: Caminho completo do arquivo zip.
        password: Senha do zip.
        extract_to: Diretório destino (default: diretório pai do zip).

    Returns:
        Lista de caminhos completos dos arquivos extraídos.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file '{zip_path}' not found.")

    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    os.makedirs(extract_to, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_names = zip_ref.namelist()
            # Tenta ler o primeiro arquivo para validar senha antes de extrair tudo
            if file_names:
                try:
                    zip_ref.open(file_names[0], pwd=password.encode("utf-8")).close()
                except RuntimeError as e:  # senha incorreta
                    raise ZipExtractionError(
                        "Senha incorreta para o arquivo zip."
                    ) from e
            zip_ref.extractall(path=extract_to, pwd=password.encode("utf-8"))
            return [os.path.join(extract_to, x) for x in file_names]
    except zipfile.BadZipFile as e:
        raise ZipExtractionError("Arquivo zip corrompido ou inválido.") from e
