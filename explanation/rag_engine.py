from learning.experience_memory import ExperienceMemory
from typing import List, Dict, Any

class RAGEngine:
    """Orchestrates retrieval of validated historical context for explanations."""
    
    def __init__(self, memory: ExperienceMemory):
        self.memory = memory

    def get_explanation_context(self, current_run_summary: str) -> List[str]:
        """Retrieves and formats similar past experiences as text snippets."""
        results = self.memory.search_similar(current_run_summary, n_results=3)
        
        snippets = []
        if results['documents']:
            for doc in results['documents'][0]:
                snippets.append(f"PAST PRECEDENT: {doc}")
                
        return snippets
