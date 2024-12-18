from bs4 import BeautifulSoup
import requests
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional

from minllm.tokenizer.bpe.base import Tokenizer


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        content_length = int(response.headers["Content-Length"])
        total_chunks = (
            (content_length // CHUNK_SIZE) + 1
            if content_length % CHUNK_SIZE != 0
            else 0
        )
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=total_chunks):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, "confirm": 1}, stream=True)
    token = get_confirm_token(response)

    content_type = response.headers["Content-Type"]
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    else:
        if content_type.startswith("text/html"):
            # Second download for large file virus warning
            html_content = response.text
            assert html_content.startswith(
                "<!DOCTYPE html><html><head><title>Google Drive - Virus scan warning"
            )

            soup = BeautifulSoup(html_content, features="html.parser")
            form_tag = soup.find("form", {"id": "download-form"})
            download_url = form_tag["action"]

            # Get all of the attributes
            id = soup.find("input", {"name": "id"})["value"]
            export = soup.find("input", {"name": "export"})["value"]
            confirm = soup.find("input", {"name": "confirm"})["value"]
            uuid = soup.find("input", {"name": "uuid"})["value"]
            params = {
                "id": id,
                "export": export,
                "confirm": confirm,
                "uuid": uuid,
            }
            response = session.get(download_url, params=params, stream=True)
    save_response_content(response, destination)


def load_dataset(
    dataset: str, context_length: int, tokenizer: Optional[Tokenizer] = None
) -> Dataset:
    assert dataset in [
        "tinyshakespeare",
        "bookscorpus",
        "race",
        "openwebtext",
        "fineweb10b",
    ]

    if dataset == "tinyshakespeare":
        from minllm.datasets import tinyshakespeare

        return tinyshakespeare.TinyShakespeareTokenized(
            ".", context_length=context_length
        )
    elif dataset == "bookscorpus":
        from minllm.datasets import bookscorpus

        return bookscorpus.BooksCorpusTokenized(
            ".", context_length=context_length, tokenizer=tokenizer
        )
    elif dataset == "openwebtext":
        from minllm.datasets import openwebtext

        return openwebtext.OpenWebTextTokenized(
            ".", context_length=context_length, tokenizer=tokenizer
        )
    elif dataset == "race":
        from minllm.datasets import race

        return race.RACETokenized(
            ".", context_length=context_length, tokenizer=tokenizer
        )
    elif dataset == "fineweb10b":
        from minllm.datasets import fineweb

        return fineweb.FineWeb10BTokenized(
            ".", context_length=context_length, tokenizer=tokenizer
        )

    assert False, f"Dataset {dataset} not found."
