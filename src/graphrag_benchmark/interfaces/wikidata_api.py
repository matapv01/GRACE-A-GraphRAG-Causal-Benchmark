import logging
import time
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class WikidataClient:
    def __init__(self, endpoint_url: str = "https://query.wikidata.org/sparql"):
        self.endpoint_url = endpoint_url
        self.agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        # Thử lại vô hạn đối với 429, 50x

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        retries = 0
        backoff = 2.0  # Initial 2 seconds backoff

        headers = {
            "User-Agent": self.agent,
            "Accept": "application/sparql-results+json, application/json",
        }

        while True:
            try:
                # To politely respect the 2 RPS rule from Wikidata
                time.sleep(0.5)
                
                response = requests.get(
                    self.endpoint_url,
                    params={"query": query},
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code in [429, 500, 502, 503, 504]:
                    retries += 1
                    
                    logger.warning(
                        f"Wikidata HTTP Error ({response.status_code}). Retrying in {backoff}s... (Attempt {retries})"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue

                response.raise_for_status()
                results = response.json()
                
                if "results" in results and "bindings" in results["results"]:
                    return results["results"]["bindings"]
                return []
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error executing SPARQL query: {e}")
                raise
            except Exception as e:
                logger.error(f"Error executing SPARQL query: {e}")
                raise
