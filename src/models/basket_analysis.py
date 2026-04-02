import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

def get_fast_cooccurrence(df: pd.DataFrame, item_col: str, group_col: str, top_n: int = 50):
    """
    Principal AI Scientist Level: Matrix-based co-occurrence with Density Filtering.
    """
    if df.empty:
        return pd.DataFrame()

    # --- NEW: DENSITY FILTERING ---
    # Filter out groups (baskets/sessions) that are too large (e.g., > 100 items)
    # These are usually store-wide catalogs, not specific "bought together" signals.
    group_sizes = df.groupby(group_col).size()
    valid_groups = group_sizes[group_sizes <= 100].index
    df = df[df[group_col].isin(valid_groups)]
    
    if df.empty:
        return pd.DataFrame()
    # ------------------------------

    # 1. Encode items and groups
    df = df.copy().dropna(subset=[item_col, group_col])
    item_categories = df[item_col].astype('category')
    group_categories = df[group_col].astype('category')
    
    item_codes = item_categories.cat.codes.values
    group_codes = group_categories.cat.codes.values
    
    # 2. Build Sparse Interaction Matrix
    ones = np.ones(len(df))
    interaction_matrix = csr_matrix((ones, (group_codes, item_codes)))
    
    # 3. Matrix Multiplication (A^T * A)
    # The Density Filter above ensures 'nnz' stays within RAM limits
    cooccurrence_sparse = interaction_matrix.T @ interaction_matrix
    
    # 4. Extract pairs
    cooccurrence_sparse.setdiag(0)
    cooccurrence_sparse.eliminate_zeros()
    
    rows, cols = cooccurrence_sparse.nonzero()
    values = cooccurrence_sparse.data
    
    mask = rows < cols
    
    results = pd.DataFrame({
        'item_a_code': rows[mask],
        'item_b_code': cols[mask],
        'co_occurrences': values[mask]
    }).sort_values('co_occurrences', ascending=False).head(top_n)
    
    names = item_categories.cat.categories
    results['Product A'] = names[results['item_a_code']].values
    results['Product B'] = names[results['item_b_code']].values
    
    return results[['Product A', 'Product B', 'co_occurrences']]