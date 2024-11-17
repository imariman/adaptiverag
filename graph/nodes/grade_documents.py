from typing import Dict, Any
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state.
    """
    print("grade documents")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score =retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade:
            print("grade document relevant")
        else:
            print("grade document is not relevant")
            web_search = True
            continue
    return {"question": question, "documents": filtered_docs, "web_search": web_search}