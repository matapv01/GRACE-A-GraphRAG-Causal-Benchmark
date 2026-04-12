from graphrag_benchmark.infrastructure.dataset_loader import DatasetLoader
from graphrag_benchmark.interfaces.wikidata_api import WikidataClient
from graphrag_benchmark.use_cases.extract_gold_path import WikidataExtractor
from graphrag_benchmark.interfaces.embedding_api import EmbeddingClient
from graphrag_benchmark.use_cases.semantic_retriever import SemanticRetriever
from rich.console import Console
from rich import print

def main():
    console = Console()
    
    console.rule("[bold green]1. Loading Dataset (Real data: LC-QuAD 2.0 with Wikidata SPARQL)")
    loader = DatasetLoader("data/lcquad_train.json")
    questions = loader.load_dataset()
    console.print(f"Loaded {len(questions)} questions with SPARQL queries.")
    for q in questions[:2]:
        console.print(f"- [yellow]ID[/yellow]: {q.id}, [cyan]Data[/cyan]: {q.question}")
        console.print(f"  [magenta]SPARQL[/magenta]: {q.sparql_query}")
        
    console.rule("[bold green]2. Extracting Gold Path (Phase 1)")
    wd_client = WikidataClient()
    wd_extractor = WikidataExtractor(wd_client)
    
    first_q = questions[0]
    console.print(f"Extracting Gold Path for: {first_q.question}")
    gold_path = wd_extractor.extract_gold_path(first_q)
    console.print(f"Gold Path Triples: {len(gold_path.triples)}")
    for t in gold_path.triples:
        console.print(f"  ({t.subject}, {t.predicate}, {t.object})")

    console.rule("[bold green]3. Beam Search Retrieval (Phase 2)")
    embed_client = EmbeddingClient()
    retriever = SemanticRetriever(wd_client, embed_client)
    
    console.print(f"Generating Candidate Reasoning Space for: {first_q.question}")
    subgraph = retriever.generate_subgraph(
        question_data=first_q,
        gold_path=gold_path,
        hop_count=1,  # limit to 1-hop for quick test
        top_k=3       # keep top 3 relations
    )
    
    console.print(f"Extra Generated Triples: {len(subgraph.extra_triples)}")
    for t in subgraph.extra_triples:
        console.print(f"  ({t.subject}, {t.predicate}, {t.object})")
    
if __name__ == "__main__":
    main()
