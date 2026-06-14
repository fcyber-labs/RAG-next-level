"""
Query rewriting using LLM to expand queries for better recall.
Handles acronyms, synonyms, and related terms.
"""

import json
import logging
import re
from typing import List

#from openai import OpenAI
from groq import Groq

from utils.metrics_exporter import export_counter   # module-level import so mocks work

logger = logging.getLogger(__name__)


def rewrite_query_with_llm(query: str, max_variations: int = 3) -> List[str]:
    """
    Rewrite query using LLM (OpenAI gpt-3.5-turbo).

    Returns a list that starts with the original query followed by up to
    `max_variations` LLM-generated alternatives.  On any failure (API error,
    invalid JSON, …) the list contains only the original query.

    Args:
        query: Original query string
        max_variations: Maximum number of LLM-generated variations to return

    Returns:
        List[str] – [original, var1, var2, …]
    """
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        #client = OpenAI()
        #response = client.chat.completions.create(
        response = client.chat.completions.create(
            model='llama3-8b-8192',
            #model='gpt-3.5-turbo',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query expansion assistant. "
                        "Return ONLY a JSON array of alternative phrasings for the user's query. "
                        "Do not include any explanation, just the JSON array."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Generate {max_variations} alternative search query variations "
                        f"for: {query}"
                    ),
                },
            ],
            max_tokens=200,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        variations = json.loads(content)

        if not isinstance(variations, list):
            raise ValueError("LLM did not return a JSON array")

        return [query] + variations[:max_variations]

    except Exception as e:
        logger.warning(f"LLM query rewriting failed, using original query: {e}")
        return [query]


def rewrite_query_rule_based(query: str) -> List[str]:
    """
    Simple rule-based query expansion (no LLM needed).

    Args:
        query: Original query

    Returns:
        List of query variations (always starts with the original)
    """
    # Common acronym mappings
    acronym_map = {
        'pto': 'paid time off',
        'ppo': 'preferred provider organization',
        'hmo': 'health maintenance organization',
        'faq': 'frequently asked questions',
        'rsu': 'restricted stock unit',
        'okr': 'objectives and key results',
        'api': 'application programming interface',
        'vpn': 'virtual private network',
        'hr': 'human resources',
        'it': 'information technology',
    }

    query_lower = query.lower()
    tokens = re.findall(r"\w+", query_lower)
    variations = [query]

    for acronym, expansion in acronym_map.items():
        if acronym in tokens:
            expanded = query_lower.replace(acronym, expansion)
            variations.append(expanded)

            # Also add OR version
            or_version = query_lower.replace(acronym, f"{acronym} OR {expansion}")
            variations.append(or_version)

    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)

    logger.info(f"Rule-based expansion: {len(unique_variations)} variations")
    return unique_variations


def rewrite_query(
    query: str,
    use_llm: bool = True,
    max_variations: int = 3,
    **kwargs,
) -> List[str]:
    """
    Main query rewriting function.

    Delegates to either the LLM or the rule-based expander.  Both sub-functions
    already include the original query as their first element, so this function
    does NOT prepend it again – it just deduplicates and exports a counter.

    Args:
        query: Original query
        use_llm: Whether to use LLM (True) or rule-based (False)
        max_variations: Max number of variations passed to the LLM expander

    Returns:
        List of unique query strings (original first, then variations)
    """
    logger.info(f"Rewriting query: '{query}' (use_llm={use_llm})")

    # Each sub-function already includes the original query
    if use_llm:
        variations = rewrite_query_with_llm(query, max_variations)
    else:
        variations = rewrite_query_rule_based(query)

    # Deduplicate while preserving order
    seen: set = set()
    unique_variations: List[str] = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)

    # Export metrics using module-level reference so unit-test mocks work
    num_rewrites = len(unique_variations) - 1   # exclude original
    export_counter('query_rewrites_total', num_rewrites)

    logger.info(f"Final query variations: {unique_variations}")
    return unique_variations