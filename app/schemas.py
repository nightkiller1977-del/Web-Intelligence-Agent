# app/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

SUPPORTED_MODES = {"quick", "standard", "deep"}
SUPPORTED_PROFILES = {"general", "technical", "repair", "code-review", "security"}
SUPPORTED_FRESHNESS_KEYS = {"since", "until", "maxAgeDays"}
SUPPORTED_INPUT_KEYS = {"documents", "repositories", "allowExternalUse"}

class LimitConfig(BaseModel):
    maximumDurationSeconds: int = Field(..., gt=0, le=3600, description="Maximum duration allowed in seconds")
    maximumSearches: int = Field(..., gt=0, le=100, description="Maximum search queries to perform")
    maximumPages: int = Field(..., gt=0, le=500, description="Maximum pages to scrape")
    maximumSources: int = Field(..., gt=0, le=100, description="Maximum sources to return in results")
    maximumMemoryMb: Optional[int] = Field(None, gt=0, le=262144, description="Per-operation memory guardrail in MB")
    maximumModelCalls: Optional[int] = Field(None, gt=0, description="Maximum estimated model calls allowed")
    maximumModelTokens: Optional[int] = Field(None, gt=0, description="Maximum estimated output tokens allowed")
    maximumModelCostUsd: Optional[float] = Field(None, gt=0, description="Maximum estimated model cost allowed in USD")

class DocumentInput(BaseModel):
    path: str = Field(..., min_length=1)
    displayName: Optional[str] = None

class RepositoryInput(BaseModel):
    path: str = Field(..., min_length=1)
    branch: Optional[str] = None

class ResearchRequestInput(BaseModel):
    operationId: str = Field(..., min_length=1, max_length=128, description="Unique operation identifier")
    idempotencyKey: Optional[str] = Field(None, min_length=1, max_length=256, description="Deterministic key to prevent duplicate runs")
    attemptId: str = Field(..., min_length=1, max_length=128, description="UUID for retry attempts")
    query: str = Field(..., min_length=1, max_length=8000, description="The user query or research topic")
    mode: str = Field("standard", description="Research mode: quick, standard, deep")
    profile: str = Field("general", description="Research profile: general, technical, repair, etc.")
    freshness: Optional[Dict[str, str]] = None
    sourcePolicy: Optional[Dict[str, Any]] = Field(None, description="Optional source policy. Only allowedDomains is supported.")
    limits: LimitConfig
    inputs: Optional[Dict[str, Any]] = None
    requireCitations: bool = True
    requireClaimVerification: Optional[bool] = Field(False, description="Return source/context passage-backed claim verification records")
    
    # Model preferences
    model_provider: Optional[str] = Field(None, description="GPT Researcher LLM provider, for example openai")
    model_name: Optional[str] = Field(None, description="GPT Researcher LLM model name")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        if value not in SUPPORTED_MODES:
            supported = ", ".join(sorted(SUPPORTED_MODES))
            raise ValueError(f"Unsupported research mode '{value}'. Supported modes: {supported}")
        return value

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        if value not in SUPPORTED_PROFILES:
            supported = ", ".join(sorted(SUPPORTED_PROFILES))
            raise ValueError(f"Unsupported research profile '{value}'. Supported profiles: {supported}")
        return value

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("freshness")
    @classmethod
    def validate_freshness(cls, value: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if value is None:
            return value
        unsupported = set(value.keys()) - SUPPORTED_FRESHNESS_KEYS
        if unsupported:
            raise ValueError(f"Unsupported freshness fields: {', '.join(sorted(unsupported))}")
        return value

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return value
        unsupported = set(value.keys()) - SUPPORTED_INPUT_KEYS
        if unsupported:
            raise ValueError(f"Unsupported inputs fields: {', '.join(sorted(unsupported))}")
        if "allowExternalUse" in value and not isinstance(value["allowExternalUse"], bool):
            raise ValueError("inputs.allowExternalUse must be a boolean")
        return value

class ResearchSource(BaseModel):
    id: str
    url: str
    title: str
    publisher: Optional[str] = None
    author: Optional[str] = None
    publishedAt: Optional[str] = None
    retrievedAt: int
    sourceType: Optional[str] = None
    qualityScore: Optional[float] = None

class ResearchEvidence(BaseModel):
    id: str
    sourceId: str
    section: Optional[str] = None
    passage: str
    contentHash: Optional[str] = None
    relevanceScore: Optional[float] = None

class ResearchClaim(BaseModel):
    id: str
    text: str
    evidenceIds: List[str]
    confidence: Optional[float] = 0.8
    verificationStatus: str = "inferred" # supported, partially-supported, unsupported, conflicting

class ResearchCitation(BaseModel):
    id: str
    sourceId: str
    evidenceIds: List[str]
    claimIds: List[str]

class ResearchMetrics(BaseModel):
    startedAt: str
    completedAt: Optional[str] = None
    durationMs: int
    researchCycles: Optional[int] = 0
    searchesPerformed: int
    pagesRead: int
    sourcesConsidered: int
    sourcesUsed: int
    modelCalls: Optional[int] = 0
    estimatedModelCostUsd: Optional[float] = 0.0

class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool
    retryAfterMs: Optional[int] = None

class ResearchResultResponse(BaseModel):
    operationId: str
    status: str # queued, running, completed, partial, failed, cancelled, interrupted
    providerId: str = "gpt-researcher"
    mode: str
    profile: str
    answer: Optional[str] = None
    sources: Optional[List[ResearchSource]] = None
    evidence: Optional[List[ResearchEvidence]] = None
    claims: Optional[List[ResearchClaim]] = None
    citations: Optional[List[ResearchCitation]] = None
    searchesPerformed: Optional[List[str]] = None
    metrics: Optional[ResearchMetrics] = None
    degraded: Optional[bool] = False
    degradedReasons: Optional[List[str]] = None
    limitations: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    requiresReconciliation: Optional[bool] = False
    error: Optional[ErrorDetail] = None

class ProgressEvent(BaseModel):
    operationId: str
    stage: str # planning, searching, reading, analyzing, follow_up, verifying, synthesizing, completed, failed
    message: str
    completedUnits: Optional[int] = None
    totalUnits: Optional[int] = None
    sourceTitle: Optional[str] = None
    sourceUrl: Optional[str] = None
    timestamp: str

class CapabilitiesInfo(BaseModel):
    service: str = "web-intelligence-agent"
    version: str
    protocol_version: str
    capabilities: Dict[str, bool]
