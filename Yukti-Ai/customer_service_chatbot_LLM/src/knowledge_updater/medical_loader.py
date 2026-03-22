"""
Load medical datasets (MedQuAD, etc.) into Document objects.
"""

import logging
from pathlib import Path
from typing import List
import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_medquad(base_path: Path) -> List[Document]:
    """
    Load all MedQuAD QA pairs from the given base path.
    Expects subfolders like 1_CancerGov_QA, 2_GARD_QA, etc., each containing CSV files.
    """
    docs = []
    if not base_path.exists():
        logger.warning(f"Medical dataset path {base_path} does not exist.")
        return docs

    # Recursively find all CSV files
    for csv_file in base_path.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            # Assume columns: question, answer, (optional) disease, etc.
            for _, row in df.iterrows():
                question = row.get('question', '')
                answer = row.get('answer', '')
                if not question or not answer:
                    continue
                # Create a document with both question and answer
                content = f"Question: {question}\nAnswer: {answer}"
                metadata = {
                    "source": str(csv_file.relative_to(base_path)),
                    "disease": row.get('disease', ''),
                    "topic": row.get('topic', '')
                }
                docs.append(Document(page_content=content, metadata=metadata))
            logger.info(f"Loaded {len(df)} QA pairs from {csv_file}")
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")
    return docs

# You can add similar loaders for other medical sources (e.g., PubMed, guidelines) later.