from typing import TypedDict


class PubMedFile(TypedDict):
    path: str
    title: str
    pmcid: str
    pmid: str
    code: str
