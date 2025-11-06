from datetime import datetime
from typing import Annotated, Optional, TypeAlias

from pydantic import BaseModel, Field

EmbeddingVector: TypeAlias = list[float]


class PaperMetadata(BaseModel):
    id: str
    title: str
    authors: list[str]
    date_published: datetime
    date_updated: datetime
    summary: str
    pdf_url: str
    embedding: EmbeddingVector


class NodeRecord(BaseModel):
    title: str | None
    description: str
    embedding: EmbeddingVector


class PaperExtractedData(BaseModel):
    content: list[NodeRecord]
    datasets: list[NodeRecord]
    models: list[NodeRecord]
    methods: list[NodeRecord]
    tasking: list[NodeRecord]


class Paper(BaseModel):
    metadata: PaperMetadata
    pdf_data: PaperExtractedData


class QueryAnswerPair(BaseModel):
    query: Annotated[str, Field(min_length=1)]
    expected_answer: Annotated[str, Field(min_length=1)]
    actual_reasoning: Annotated[
        str,
        Field(
            description="The reasoning process including traversal hops in format: 'Node1 (Type)' -> 'Node2 (Type)' -> 'Node3 (Type)'",
        ),
    ]
    actual_answer: Annotated[
        str, Field(
            min_length=1, description="Natural language answer to the question")
    ]
    actual_evidence: Optional[list[str]] = Field(
        default=None,
        description="List of retrieved text snippets that support the answer"
    )


class GroundednessCheckModel(BaseModel):
    support_claims: Annotated[
        int, Field(
            description="Number of sentences supported by any evidence snippet")
    ]
    total_claims: Annotated[int, Field(
        description="Total number of sentences")]
    grounded_ratio: Annotated[
        float, Field(description="Ratio of grounded claims to total claims")
    ]
    unsupported_examples: Annotated[
        list[str], Field(
            description="List of unsupported sentences from the answer")
    ]


class RelevanceCheckModel(BaseModel):
    score: Annotated[float, Field(ge=0, le=1, description="Relevance score")]
    reasoning: Annotated[
        str,
        Field(
            description="Reasoning for the relevance score",
        ),
    ]


class CompletenessCheckModel(BaseModel):
    score: Annotated[float, Field(ge=0, le=1, description="Relevance score")]
    missing: Annotated[list[str], Field(
        description="List of missing keywords")]


class AccuracyCheckModel(BaseModel):
    is_accurate: Annotated[bool, Field(
        description="Whether the actual answer matches the expected answer")]
    reasoning: Annotated[str, Field(
        description="Explanation of why the answer is or isn't accurate")]


class AccuracyCheckLLMModel(BaseModel):
    """LLM-based semantic accuracy check result"""
    is_accurate: Annotated[bool, Field(
        description="Whether the actual answer semantically matches the expected answer")]
    confidence: Annotated[float, Field(
        ge=0, le=1, description="Confidence score in the accuracy judgment (0-1)")]
    reasoning: Annotated[str, Field(
        description="Detailed explanation of semantic comparison including paraphrasing, synonyms, or mismatches")]


class QAEvaluationModel(BaseModel):
    groundedness_check: GroundednessCheckModel
    relevance_check: RelevanceCheckModel
    completeness_check: CompletenessCheckModel
    accuracy_check: AccuracyCheckModel
    accuracy_check_llm: Optional[AccuracyCheckLLMModel] = Field(
        default=None,
        description="Optional LLM-based semantic accuracy check"
    )


class QAResultModel(BaseModel):
    qapair: QueryAnswerPair
    evaluation: QAEvaluationModel


class AblationConfig(BaseModel):
    """
    Configuration for ablation studies to control GraphRAG behavior.

    Examples:
        # Baseline: No GraphRAG at all
        AblationConfig(enable_graphrag=False)

        # Only vector search, no graph traversal
        AblationConfig(enable_cypher_queries=False)

        # Only graph queries, no vector search
        AblationConfig(enable_vector_search=False)

        # Exclude Dataset and Content nodes (only Paper and Author remain)
        AblationConfig(excluded_node_types=["Dataset", "Content"])

        # Exclude specific relationships
        AblationConfig(excluded_relationships=["HAS_CONTENT"])
    """
    enable_graphrag: bool = True  # Master switch: False = no tools available at all
    enable_vector_search: bool = True  # Control vector_search tool
    enable_cypher_queries: bool = True  # Control query_neo4j tool
    # Node types to exclude (e.g., ["Content", "Dataset"])
    excluded_node_types: Optional[list[str]] = None
    # Relationships to exclude from queries
    excluded_relationships: Optional[list[str]] = None
    # Override top_k for vector search
    max_vector_results: Optional[int] = 5
