from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from rdflib import Graph, URIRef, Literal
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG Benchmark - Sandbox SPARQL Endpoint")

# In-memory graph state for the current evaluation context
current_graph = Graph()

class TripleModel(BaseModel):
    subject: str
    predicate: str
    object: str

@app.post("/admin/load_triples")
def load_triples(triples: List[TripleModel]):
    """
    Admin endpoint to reset and load a specific Subgraph (Clean or Perturbed) into memory.
    This effectively isolates the environment for the framework being evaluated.
    """
    global current_graph
    current_graph = Graph() # Đặt lại RAM
    
    for t in triples:
        s = URIRef(t.subject) if t.subject.startswith("http") else URIRef(f"http://example.org/{t.subject}")
        p = URIRef(t.predicate) if t.predicate.startswith("http") else URIRef(f"http://example.org/{t.predicate}")
        o = URIRef(t.object) if str(t.object).startswith("http") else Literal(t.object)
        
        current_graph.add((s, p, o))
        
    return {"message": f"Successfully loaded {len(triples)} triples into the Sandbox DB."}

@app.get("/sparql")
@app.post("/sparql")
async def sparql_endpoint(request: Request):
    """
    Mocking Standard Wikidata SPARQL Endpoint.
    GNN-RAG or ToG will send queries here, but they will only see the isolated subgraph.
    """
    query = ""
    if request.method == "GET":
        query = request.query_params.get("query", "")
    else:
        ctype = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in ctype:
            form = await request.form()
            query = form.get("query", "")
        elif "application/sparql-query" in ctype:
            body = await request.body()
            query = body.decode()
        else:
            body = await request.json()
            query = body.get("query", "")
            
    if not query:
        raise HTTPException(status_code=400, detail="Missing query parameter")
        
    try:
        results = current_graph.query(query)
        bindings = []
        vars_list = [str(v) for v in results.vars] if results.vars else []
        
        for row in results:
            binding = {}
            for var, val in zip(vars_list, row):
                if val is not None:
                    val_str = str(val)
                    type_str = "uri" if isinstance(val, URIRef) else "literal"
                    binding[var] = {"type": type_str, "value": val_str}
            bindings.append(binding)
            
        return JSONResponse(content={
            "head": {"vars": vars_list},
            "results": {"bindings": bindings}
        })
    except Exception as e:
        logger.error(f"Failed to execute SPARQL in sandbox Context: {e}")
        raise HTTPException(status_code=400, detail=str(e))
