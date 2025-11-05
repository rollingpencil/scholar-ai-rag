import hashlib
import json
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models.models import AblationConfig
from service.embed_svc import embed_content
from service.neo4j_svc import get_graph_schema, process_query, process_vector_search
from utils.logger import log
from utils.utils import get_envvar

CACHE_ENABLE: bool = bool(get_envvar("QUERY_CACHE_ENABLE"))
CACHE_TTL: int = int(get_envvar("QUERY_CACHE_TTL"))


def _get_base_prompt_rules() -> str:
    """Common rules for all prompt variations."""
    return """
CRITICAL STOPPING RULES - YOU MUST FOLLOW THESE:
1. Maximum 5 tool calls total per question - after that you MUST return your answer with whatever information you have
2. After EVERY tool call, ask yourself: "Do I have enough information to answer the question?" If YES, STOP and return the answer immediately
3. If a tool returns empty results, STOP trying variations - return an answer stating no information was found
4. If a tool returns ANY relevant data, use it to formulate an answer - do NOT keep searching for "better" results
5. NEVER call the same tool with the same or similar parameters more than once

OUTPUT FORMAT REQUIREMENTS:
- 'reasoning': Show your reasoning process. If using graph traversal, show path as "Node1 (Type)" -> "Node2 (Type)"
- 'answer': Natural language answer to the question
- If no information found, clearly state that in the answer

WHAT NOT TO DO (violations will waste resources):
- ❌ Making more than 5 tool calls
- ❌ Retrying queries with slightly different parameters
- ❌ Calling tools "just to be thorough" when you already have an answer
- ❌ Making separate queries for each piece of information instead of one comprehensive query
- ❌ Continuing to search after finding relevant information
"""

NEO4J_QUERY_SYSTEM_PROMPT_FULL = f"""
You are a Neo4j graph database expert with access to both Cypher queries and vector similarity search.

{get_graph_schema()}

{_get_base_prompt_rules()}

IMPORTANT KNOWLEDGE ABOUT THE GRAPH STRUCTURE:
- Paper nodes have: id, title, summary, date_published, date_updated, embedding
- Author nodes ONLY have: name (NO affiliation or other details)
- Content nodes contain the actual paper text chunks with: description (the text), embedding
- Relationship: Paper -[CONTAINS_CHUNK]-> Content
- **Detailed information (like author affiliations, specific facts, quotes) is ONLY in Content chunks, NOT in graph node properties**

AVAILABLE VECTOR INDEXES:
- content_embedding_index (search Content chunks for detailed information)
- paper_vector_index (search papers by topic/summary)
- dataset_vector_index, method_vector_index, model_vector_index, task_vector_index

QUERY STRATEGY (follow in order):
Step 1: Determine what type of query you need:
   a) **For questions about author affiliations, detailed facts, or specific information from paper content:**
      - Use vector_search on 'content_embedding_index' with your specific question to find relevant Content chunks
      - The answer will be in the Content node's 'description' field
      - Example: "What is the affiliation of author X in paper Y?" → vector_search("affiliation of Aditi Singh paper 2501.09136v3", "content_embedding_index", top_k=5)
      - Example: "Which author is from The Davey Tree Expert Company?" → vector_search("The Davey Tree Expert Company author affiliation", "content_embedding_index", top_k=5)

   b) **For simple graph structure queries with EXACT entity names/IDs:**
      - If you have the exact Paper ID, use query_neo4j with exact matches
      - Example: MATCH (p:Paper {{id: "2501.09136v3"}})-[:WRITTEN_BY]->(a:Author) RETURN a.name

   c) **For semantic queries without exact matches (fuzzy/similar entities):**
      - Use HYBRID approach: vector_search → graph traversal
      - First: Use vector_search to find semantically similar starting nodes
      - Then: Use query_neo4j to traverse from those nodes
      - Example: "Papers about RAG" → vector_search("Retrieval Augmented Generation", "paper_vector_index")
                → Get paper IDs → MATCH (p:Paper)-[:USES_METHOD]->(m) WHERE p.id IN [ids] RETURN p, m

   d) **For semantic paper/dataset/method/model/task discovery:**
      - Use vector_search on the appropriate index (paper_vector_index, dataset_vector_index, etc.)
      - If you need to traverse from those results, follow with query_neo4j

Step 2: For complex queries combining semantic search + graph traversal:
   - Pattern: vector_search (find starting nodes) → query_neo4j (traverse relationships)
   - Extract IDs/titles from vector_search results
   - Use those in WHERE clauses: WHERE p.id IN ["id1", "id2", ...] or WHERE m.title IN ["title1", "title2"]
   - This allows finding similar entities (not exact matches) then exploring their relationships

Step 3: STOP and return your answer with the format above.

Remember: Vector indexes enable FUZZY matching (similar but not exact). Use them to find starting nodes, then traverse!
"""

NEO4J_QUERY_SYSTEM_PROMPT_VECTOR_ONLY = f"""
You are a semantic search expert with access to vector similarity search over a Neo4j graph database.

{get_graph_schema()}

{_get_base_prompt_rules()}

IMPORTANT KNOWLEDGE ABOUT THE GRAPH STRUCTURE:
- Paper nodes have: id, title, summary (but NOT full content)
- Author nodes ONLY have: name (NO affiliation or other details)
- Content nodes contain the actual paper text chunks with: description (the text), embedding
- **Detailed information (like author affiliations, specific facts, quotes) is ONLY in Content chunks via 'content_embedding_index'**

AVAILABLE VECTOR INDEXES:
- content_embedding_index (search Content chunks for detailed information - USE THIS for affiliations, facts, quotes)
- paper_vector_index (search papers by topic/summary)
- dataset_vector_index, method_vector_index, model_vector_index, task_vector_index

QUERY STRATEGY:
Step 1: Determine which index to search:
   - For detailed information about paper content, author affiliations, specific facts → Use 'content_embedding_index'
   - For finding papers by semantic topic/theme (not exact title) → Use 'paper_vector_index'
   - For finding datasets, models, methods, tasks by semantic similarity → Use their respective vector indexes
   - Note: Vector search finds SIMILAR entities, not just exact matches (e.g., "RAG" finds "Retrieval-Augmented Generation")

Step 2: Use vector_search with a well-crafted semantic query
   - Choose the appropriate index based on Step 1
   - Set top_k appropriately (typically 3-10)
   - For author affiliations: vector_search("affiliation of [author name] [paper context]", "content_embedding_index", top_k=5)
   - For semantic entity discovery: vector_search("Retrieval Augmented Generation systems", "paper_vector_index", top_k=5)
   - Results include similarity scores - higher scores = more relevant

Step 3: Analyze results and determine if you need more context:
   - If results have enough information, STOP and answer
   - If you need to explore relationships (e.g., "what methods do these papers use?"), you'll need Cypher queries (not available in vector-only mode)
   - Note: In vector-only mode, you can't traverse graph relationships - only get node properties

Step 4: STOP and return your answer based on the vector search results
   - Extract information from node properties (especially 'description' field for Content nodes)
   - Summarize the information found in the 'answer' field
   - Explain what you found in the 'reasoning' field

Remember: Vector indexes find SIMILAR entities (fuzzy matching). You only have vector search - no graph traversal available!
"""

NEO4J_QUERY_SYSTEM_PROMPT_CYPHER_ONLY = f"""
You are a Neo4j Cypher query expert with access to a graph database.

{get_graph_schema()}

{_get_base_prompt_rules()}

IMPORTANT KNOWLEDGE ABOUT THE GRAPH STRUCTURE:
- Paper nodes have: id, title, summary, date_published, date_updated, embedding
- Author nodes ONLY have: name (NO affiliation or other details)
- Content nodes contain the actual paper text chunks with: description (the text), embedding
- Relationship: Paper -[CONTAINS_CHUNK]-> Content
- **Detailed information (like author affiliations, specific facts) is in Content node 'description' fields, NOT in Author nodes**
- **NOTE: Vector search is disabled, so you must use Cypher text matching (CONTAINS, regex) on Content.description**

QUERY STRATEGY:
Step 1: Determine your approach:
   a) **For simple graph queries (authors, papers, relationships):**
      - Write a direct Cypher query with exact matches
      - Example: MATCH (p:Paper {{id: "..."}})-[:WRITTEN_BY]->(a:Author) RETURN a.name

   b) **For detailed information (affiliations, specific facts):**
      - Query Content chunks and search their 'description' fields using text matching
      - Example: MATCH (p:Paper {{id: "2501.09136v3"}})-[:CONTAINS_CHUNK]->(c:Content)
                 WHERE c.description CONTAINS "Aditi Singh" AND (c.description CONTAINS "affiliation" OR c.description CONTAINS "Department" OR c.description CONTAINS "University")
                 RETURN c.description LIMIT 5
      - Use case-insensitive regex for better matching: WHERE c.description =~ "(?i).*affiliation.*"
      - Combine multiple keywords with AND/OR for better results

Step 2: Write ONE comprehensive Cypher query that gets all the information you need
   - Use OPTIONAL MATCH for relationships that might not exist
   - Use WHERE clauses with CONTAINS or regex for text matching in Content descriptions
   - Return all necessary data in a single query including Content.description for detailed info
   - LIMIT results to avoid returning too much data

Step 3: STOP and return your answer based on the query results
   - Show the traversal path in 'reasoning' if applicable
   - Extract relevant details from Content.description fields
   - Provide natural language answer in 'answer' field

Remember: No vector search available. Query Content nodes with CONTAINS/regex to find affiliations and facts!
"""

NEO4J_QUERY_SYSTEM_PROMPT_NO_TOOLS = f"""
You are a helpful assistant that answers questions about academic papers and research.

{_get_base_prompt_rules()}

IMPORTANT: You do NOT have access to any database or tools for this query.

QUERY STRATEGY:
- Answer based solely on general knowledge
- Be honest if you don't have specific information
- State clearly in your reasoning that no database access was available
- Provide the best general answer you can in the 'answer' field

Remember: No tools available - provide your answer immediately based on general knowledge.
"""


class GraphQueryResult(BaseModel):
    reasoning: str = Field(
        description="The reasoning process including traversal hops in format: 'Node1 (Type)' -> 'Node2 (Type)' -> 'Node3 (Type)'"
    )
    answer: str = Field(description="Natural language answer to the question")


_query_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_vector_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}

log.info(f"Query Model set: {get_envvar('QUERY_MODEL_NAME')}")
llm_model = OpenAIChatModel(
    get_envvar("QUERY_MODEL_NAME"),
    provider=OpenAIProvider(
        base_url=get_envvar("OPENAI_COMPAT_API_ENDPOINT"),
        api_key=get_envvar("OPENAI_COMPAT_API_KEY"),
    ),
)

def _create_query_agent(config: AblationConfig) -> Agent:
    """Create an agent with tools based on ablation configuration."""
    # Select the appropriate system prompt based on configuration
    if not config.enable_graphrag:
        system_prompt = NEO4J_QUERY_SYSTEM_PROMPT_NO_TOOLS
    elif not config.enable_vector_search and not config.enable_cypher_queries:
        system_prompt = NEO4J_QUERY_SYSTEM_PROMPT_NO_TOOLS
    elif not config.enable_vector_search:
        system_prompt = NEO4J_QUERY_SYSTEM_PROMPT_CYPHER_ONLY
    elif not config.enable_cypher_queries:
        system_prompt = NEO4J_QUERY_SYSTEM_PROMPT_VECTOR_ONLY
    else:
        system_prompt = NEO4J_QUERY_SYSTEM_PROMPT_FULL

    # Create agent
    agent = Agent(
        llm_model,
        system_prompt=system_prompt,
        output_type=GraphQueryResult,
        retries=int(get_envvar("QUERY_RETRIES")),
    )

    # Only register tools that are enabled
    if config.enable_graphrag:
        if config.enable_cypher_queries:
            agent.tool(query_neo4j)
        if config.enable_vector_search:
            agent.tool(vector_search)

    return agent


async def query(question: str, ablation_config: Optional[AblationConfig] = None) -> GraphQueryResult:
    """
    Query Neo4j using natural language (async).

    Args:
        question: Natural language question
        ablation_config: Optional configuration for ablation studies to control GraphRAG behavior

    Returns:
        GraphQueryResult with answer and data
    """
    # Store ablation config in a global for tools to access
    global _current_ablation_config
    _current_ablation_config = ablation_config or AblationConfig()

    # Log ablation configuration
    if ablation_config:
        log.info(f"Ablation Config: {ablation_config.model_dump()}")

    # Create agent with appropriate tools based on config
    agent = _create_query_agent(_current_ablation_config)

    result = await agent.run(
        question,
        model_settings=json.loads(get_envvar("QUERY_MODEL_SETTINGS")),
        usage_limits=UsageLimits(request_limit=100)  # Increase limit from default 50 to 100
    )
    return result.output


# Global to store current ablation config
_current_ablation_config: AblationConfig = AblationConfig()


def query_neo4j(ctx: RunContext[None], cypher_query: str) -> list[dict[str, Any]]:
    """
    Execute a Cypher query against the Neo4j database.

    Args:
        cypher_query: The Cypher query to execute

    Returns:
        List of records as dictionaries
    """
    # Filter query based on excluded node types and relationships
    modified_query = _apply_ablation_filters(cypher_query)
    if modified_query != cypher_query:
        log.info(
            f"Query modified by ablation config:\nOriginal: {cypher_query}\nModified: {modified_query}")

    # Clean expired cache entries periodically
    _clean_expired_cache()

    # Check cache first
    cache_func = None
    if CACHE_ENABLE:
        cache_key = _get_cache_key(modified_query)
        if cache_key in _query_cache:
            cached_result, timestamp = _query_cache[cache_key]
            if _is_cache_valid(timestamp):
                log.debug(f"\n{'=' * 60}")
                log.debug("CACHE HIT - Using cached result")
                log.debug(f"{'=' * 60}")
                log.debug(f"Query: {modified_query}")
                log.debug(f"{'=' * 60}\n")
                return cached_result
        cache_func = _query_cache_func

    log.debug(f"{'=' * 60}")
    log.debug("GENERATED CYPHER QUERY:")
    log.debug(f"{'=' * 60}")
    log.debug(modified_query)
    log.debug(f"{'=' * 60}\n")

    return process_query(modified_query, cache_func=cache_func)


def _query_cache_func(cypher_query, records):
    log.debug(f"{'=' * 60}")
    log.debug("QUERY CACHE MISS - Caching result")
    log.debug(f"{'=' * 60}")
    cache_key = _get_cache_key(cypher_query)
    _query_cache[cache_key] = (records, time.time())


async def vector_search(
    ctx: RunContext[None], query_text: str, index_name: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """
    Search for nodes using vector similarity on their embeddings.

    Args:
        query_text: The text to search for semantically (e.g., "transforms unstructured PDFs into a queryable ontology")
        index_name: Name of the vector index to search. Available indexes are shown in the schema.
        top_k: Number of top results to return (default 5)

    Returns:
        List of matching nodes with their properties and similarity scores
    """
    # Apply max_vector_results override if specified
    if _current_ablation_config.max_vector_results is not None:
        top_k = min(top_k, _current_ablation_config.max_vector_results)
        log.info(f"Vector search top_k limited to {top_k} by ablation config")

    # Clean expired cache entries periodically
    _clean_expired_cache()

    # Check cache first
    cache_func = None
    cache_query = f"{query_text}|{index_name}|{top_k}"
    if CACHE_ENABLE:
        cache_key = _get_cache_key(cache_query)
        if cache_key in _vector_cache:
            cached_result, timestamp = _vector_cache[cache_key]
            if _is_cache_valid(timestamp):
                log.debug(f"\n{'=' * 60}")
                log.debug("CACHE HIT - Using cached vector search result")
                log.debug(f"{'=' * 60}")
                log.debug(f"Query: {query_text}")
                log.debug(f"Index: {index_name}")
                log.debug(f"Top K: {top_k}")
                log.debug(f"{'=' * 60}\n")
                # Apply node type filtering to cached results
                return _filter_results_by_node_type(cached_result)
        cache_func = _vector_search_cache_func

    log.debug(f"{'=' * 60}")
    log.debug("VECTOR SEARCH:")
    log.debug(f"{'=' * 60}")
    log.debug(f"Query: {query_text}")
    log.debug(f"Index: {index_name}")
    log.debug(f"Top K: {top_k}")
    log.debug(f"{'=' * 60}\n")

    # Generate embedding for the query text
    query_embedding = await embed_content(query_text)
    log.debug(f"Generated embedding (dim: {len(query_embedding)})")

    # Execute the vector search query
    results = process_vector_search(
        index_name, top_k, query_embedding, query_text, cache_func=cache_func
    )

    # Apply node type filtering
    return _filter_results_by_node_type(results)


def _vector_search_cache_func(cache_query, records):
    log.debug(f"{'=' * 60}")
    log.debug("VECTOR CACHE MISS - Caching result")
    log.debug(f"{'=' * 60}")
    cache_key = _get_cache_key(cache_query)
    _vector_cache[cache_key] = (records, time.time())


def _get_cache_key(query: str) -> str:
    """Generate a cache key from a query string"""
    return hashlib.md5(query.encode()).hexdigest()


def _is_cache_valid(timestamp: float) -> bool:
    """Check if a cached entry is still valid"""
    return (time.time() - timestamp) < CACHE_TTL


def _clean_expired_cache():
    """Remove expired entries from both caches"""
    current_time = time.time()

    # Clean query cache
    expired_keys = [
        key
        for key, (_, timestamp) in _query_cache.items()
        if (current_time - timestamp) >= CACHE_TTL
    ]
    for key in expired_keys:
        del _query_cache[key]

    # Clean vector cache
    expired_keys = [
        key
        for key, (_, timestamp) in _vector_cache.items()
        if (current_time - timestamp) >= CACHE_TTL
    ]
    for key in expired_keys:
        del _vector_cache[key]


def clear_cache():
    """Manually clear all caches"""
    _query_cache.clear()
    _vector_cache.clear()
    log.debug("Cache cleared")


def _apply_ablation_filters(cypher_query: str) -> str:
    """
    Apply ablation config filters to a Cypher query.

    This function modifies the query to exclude certain node types and relationships
    based on the current ablation configuration.

    Args:
        cypher_query: Original Cypher query

    Returns:
        Modified Cypher query with ablation filters applied
    """
    modified_query = cypher_query

    # Add WHERE clauses to exclude certain node types
    if _current_ablation_config.excluded_node_types:
        for node_type in _current_ablation_config.excluded_node_types:
            # Simple approach: add a global WHERE clause
            if "WHERE" in modified_query:
                # Find all variable names used in MATCH clauses
                variables = re.findall(r'MATCH\s+\((\w+)', modified_query)
                for var in set(variables):
                    where_clause = f" AND NOT {var}:{node_type}"
                    if where_clause not in modified_query:
                        modified_query = modified_query.replace(
                            " RETURN", f"{where_clause} RETURN", 1)
            else:
                # Add new WHERE clause before RETURN
                variables = re.findall(r'MATCH\s+\((\w+)', modified_query)
                if variables:
                    where_conditions = [
                        f"NOT {var}:{node_type}" for var in set(variables)]
                    where_clause = f" WHERE {' AND '.join(where_conditions)}"
                    modified_query = modified_query.replace(
                        " RETURN", f"{where_clause} RETURN", 1)

    # Filter out excluded relationships
    if _current_ablation_config.excluded_relationships:
        for rel_type in _current_ablation_config.excluded_relationships:
            pattern = f'\\[\\w*:{rel_type}\\]'
            if re.search(pattern, modified_query):
                log.warning(
                    f"Query contains excluded relationship type {rel_type}, returning empty result")
                # Rather than trying to remove it, just flag it
                # You might want to return an empty result or modify the query differently

    return modified_query


def _filter_results_by_node_type(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter vector search results based on excluded node types.

    Args:
        results: List of nodes from vector search

    Returns:
        Filtered list with excluded node types removed
    """
    if not _current_ablation_config.excluded_node_types:
        return results

    filtered = []
    for result in results:
        # Check if the node has labels
        node = result.get('node', {})
        labels = node.get('labels', [])

        # If node is actually in the result dict itself (different structure)
        if not labels and 'labels' in result:
            labels = result['labels']

        # Check if any of the node's labels are in the excluded list
        is_excluded = any(
            label in _current_ablation_config.excluded_node_types for label in labels)

        if not is_excluded:
            filtered.append(result)
        else:
            log.debug(
                f"Filtered out node with labels {labels} due to ablation config")

    return filtered
