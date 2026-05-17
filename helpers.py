import pandas as pd
import math

def extract_logprob_table(response, top_k=5):
    """
    Extracts a ranked table of the generated token + top logprob alternatives.
    
    Args:
        response: OpenAI/LiteLLM response object
        top_k: number of rows to return (default 5)
    
    Returns:
        pandas DataFrame with rank, token, logprob, probability
    """
    lp = response.choices[0].logprobs.content
    item = lp[0]

    candidates = item.top_logprobs or []

    candidates = [t for t in candidates if t.token != item.token]

    rows = [{
        "token": item.token,
        "logprob": item.logprob,
        "probability": math.exp(item.logprob)
    }]

    rows += [
        {
            "token": t.token,
            "logprob": t.logprob,
            "probability": math.exp(t.logprob)
        }
        for t in candidates
    ]

    rows = rows[:top_k]

    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))

    return df