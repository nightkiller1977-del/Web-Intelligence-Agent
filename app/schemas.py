# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class LimitConfig(BaseModel):
    maximumDurationSeconds: int = Field(..., description="Maximum duration allowed in seconds")
    maximumSearches: int = Field(..., description="Maximum search queries to perform")
    maximumPages: int = Field(..., description="Maximum pages to scrape")
    maximumSources: int = Field(..., description="Maximum sources to return in results")
    maximumModelCalls: Optional[int] = None
    maximumModelTokens: Optional[int] = None
    maximumModelCostUsd: Optional[float] = None

class DocumentInput(BaseModel):
    path: str
    displayName: Optional[str] = None

class RepositoryInput(BaseModel):
    path: str
    branch: Optional[str] = None

class ResearchRequestInput(BaseModel):
    operationId: str = Field(..., description="Unique operation identifier")
    idempotencyKey: str = Field(..., description="Deterministic key to prevent duplicate runs")
    attemptId: str = Field(..., description="UUID for retry attempts")
    query: str = Field(..., description="The user query or research topic")
    mode: str = Field("standard", description="Research mode: quick, standard, deep")
    profile: str = Field("general", description="Research profile: general, technical, repair, etc.")
    freshness: Optional[Dict[str, str]] = None
    sourcePolicy: Optional[Dict[str, Any]] = None
    limits: LimitConfig
    inputs: Optional[Dict[str, Any]] = None
    requireCitations: bool = True
    requireClaimVerification: Optional[bool] = False
    
    # Model preferences
    model_provider: Optional[str] = None
    model_name: Optional[str] = None

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
    status: str # completed, partial, failed, cancelled, interrupted
    providerId: str = "gpt-researcher"
    mode: str
    profile: str
    answer: str
    sources: List[ResearchSource]
    evidence: List[ResearchEvidence]
    claims: List[ResearchClaim]
    citations: List[ResearchCitation]
    searchesPerformed: List[str]
    metrics: ResearchMetrics
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
