import logging
import time
from urllib.error import HTTPError
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class WikidataClient:
    def __init__(self, endpoint_url: str = "https://query.wikidata.org/sparql"):
        self.endpoint_url = endpoint_url
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        # Using a custom User-Agent to comply with Wikidata's policy
        self.sparql.agent = "GraphRAG-Benchmark/1.0 (https://github.com/MinhPV/GraphRAG-Benchmark; mailto:minhpv@ptit.edu.vn)"
        self.max_retries = 5

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        self.sparql.setQuery(query)
        retries = 0
        backoff = 2.0  # Initial 2 seconds backoff
        
        while True:
            try:
                # To politely respect the 2 RPS rule from Wikidata
                time.sleep(0.5) 
                results = self.sparql.query().convert()
                if "results" in results and "bindings" in results["results"]:
                    return results["results"]["bindings"]
                return []
            except HTTPError as e:
                if e.code == 429:
                    retries += 1
                    logger.warning(f"Wikidata Rate Limit (429). Retrying in {backoff}s... (Attempt {retries})")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)  # Tăng dần nhưng tối đa đợi 60s rồi thử lại
                else:
                    logger.error(f"Error executing SPARQL query: {e}")
                    raise
            except Exception as e:
                logger.error(f"Error executing SPARQL query: {e}")
                raise
