"""All pipeline tunables in one place."""

from pathlib import Path

# Resolve paths relative to the section root so the pipeline works no matter
# which directory it is launched from.
SECTION_ROOT = Path(__file__).resolve().parent.parent

CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 120
# 8 of ~25 chunks: generous on purpose. Cross-document questions (e.g. cash
# refund rules split between refund_policy and payments_and_billing) need the
# supporting chunks that rank 5th-8th, and the LLM grader — not k — is the
# relevance gate. At production corpus sizes this would be retrieve-more +
# re-rank instead (see NOTES.md).
TOP_K = 8
# Spec called for gemini-2.0-flash, but the free tier now has zero quota for
# it (limit: 0), and gemini-2.5-flash is retired for new users. Among current
# models, gemini-3.1-flash-lite is stable (non-preview), fast/cheap with good
# JSON adherence, and has a comfortable free-tier daily quota (the larger
# gemini-3.5-flash free-tier daily cap is very small).
LLM_MODEL = "gemini-3.1-flash-lite"
# Spec called for models/text-embedding-004, but Google retired it (the API
# now returns 404 for it). gemini-embedding-001 is its stable GA successor
# with the same retrieval_document/retrieval_query task types.
EMBEDDING_MODEL = "models/gemini-embedding-001"
INDEX_DIR = SECTION_ROOT / "storage" / "faiss_index"
DATA_DIR = SECTION_ROOT / "data"
ABSTAIN_MESSAGE = (
    "I couldn't find information about this in the support documents. "
    "This question may be outside the scope of what I can answer reliably."
)

# Retry policy for LLM calls. The Gemini free tier allows only a handful of
# requests per minute, and every rejected retry still counts against that
# quota — so retry sparsely with long waits (the API's own retry hints are
# typically 30-60s) rather than hammering with short exponential backoff.
LLM_MAX_RETRIES = 4
LLM_RETRY_BASE_DELAY = 20.0  # seconds; doubles per attempt (20, 40, 80)
LLM_INNER_RETRIES = 2  # langchain-google-genai's own tenacity retries
