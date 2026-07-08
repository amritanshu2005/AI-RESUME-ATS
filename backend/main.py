import logging
from contextlib import asynccontextmanager
import hashlib
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import(
    ALLOWED_ORIGINS, 
    APP_DESCRIPTION, 
    APP_TITLE, 
    APP_VERSION, 
    SPACY_MODEL_PRIMARY, 
    SPACY_MODEL_SECONDARY, SENTENCE_TRANSFORMER_MODEL
)
from backend.api.routes import router

logger=logging.getLogger('ats_resume_scorer')


class _FallbackEmbedder:
    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def encode(self, sentences, convert_to_tensor=False):
        if isinstance(sentences, str):
            return self._encode_one(sentences)
        return [self._encode_one(sentence) for sentence in sentences]

    def _encode_one(self, sentence: str):
        vector = [0.0] * self.dimension
        for token in sentence.lower().split():
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            index = digest[0] % self.dimension
            weight = (int.from_bytes(digest[1:5], 'big') % 1000) / 1000.0
            vector[index] += weight
        return vector


def _load_spacy_model():
    import spacy

    for model_name in (SPACY_MODEL_PRIMARY, SPACY_MODEL_SECONDARY):
        try:
            logger.info(f'Loading spaCy NLP model: {model_name}')
            nlp = spacy.load(model_name)
            logger.info(f'Loaded {model_name}')
            return nlp
        except Exception as exc:
            logger.warning(f'Could not load spaCy model {model_name}: {exc}')

    logger.warning('Using blank spaCy English pipeline fallback')
    return spacy.blank('en')


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f'Loading SentenceTransformer: {SENTENCE_TRANSFORMER_MODEL}')
        embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        logger.info(f'Loaded {SENTENCE_TRANSFORMER_MODEL}')
        return embedder
    except Exception as exc:
        logger.warning(f'Could not load SentenceTransformer {SENTENCE_TRANSFORMER_MODEL}: {exc}')
        logger.warning('Using fallback hashed embedder')
        return _FallbackEmbedder()

@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info('Starting ATS Resume Analyzer API...')

    app.state.nlp = _load_spacy_model()
    app.state.embedder = _load_embedder()

    logger.info('All models loaded. API is ready to serve requests.')

    yield

    logger.info('shutting down the api!!')

app=FastAPI(
    title=APP_TITLE, 
    description=APP_DESCRIPTION, 
    version=APP_VERSION, 
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc'
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True, 
    allow_methods     = ['*'],
    allow_headers     = ['*'],

)

app.include_router(router)

@app.get('/')
async def root():
    return {
        'name':      'ATS Resume Analyzer API',
        'version':   '2.0.0',
        'endpoints': {
            'POST   /api/v1/analyze-resume': 'Analyze a resume',
            'GET    /api/v1/history':        'Get user history',
            'DELETE /api/v1/history/:id':    'Delete a history entry',
            'GET    /api/v1/health':         'Health check',
            'POST   /api/v1/generate-pdf':   'Generate PDF report from data',
        },
    }

if __name__=='__main__':
    import uvicorn
    uvicorn.run(
        'backend.main:app',
        host    = '0.0.0.0',
        port    = 8000,
        reload  = True,    # Auto-restart on code changes (dev only)
    )
