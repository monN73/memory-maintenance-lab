"""Safe filesystem identities for model-specific experiment artifacts."""
from urllib.parse import quote


def model_directory(model: str) -> str:
    if not model or model in ('.', '..'):
        raise ValueError('Model name must be nonempty and cannot be . or ..')
    return quote(model, safe='')
