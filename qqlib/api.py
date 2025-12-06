from dataclasses import dataclass

from qqlib.api_client import APIClient
from qqlib.api_registry import APIRegistry

@dataclass
class APIServices:
    registry: APIRegistry = None
    client: APIClient = None

    def __post_init__(self):
        if self.registry is None:
            self.registry = APIRegistry()
        if self.client is None:
            self.client = APIClient(self.registry)
