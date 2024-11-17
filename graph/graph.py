from graph.node_consts import RETRIEVE, GENERATE, WEBSEARCH, GRADE_DOCUMENTS
from graph.nodes import generate, grade_documents, web_search, retrieve
from graph.chains.router import RouteQuery, question_router
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv

load_dotenv()

def decide_to_generate(state: GraphState):
    print("decide to generate")
    if state["web_search"]:
        return WEBSEARCH
    else:
        return GENERATE

def grade_generation_grounded_in_documents_and_questions(state: GraphState) -> str:
    print("check hallucilation")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grade := score.binary_score:
        print("Generation is grounded in documents")
        score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_grade := score.binary_score:
            print("Generation addresses question")
            return "useful"
        else:
            print("Generation does not address question")
            return "not useful"
    else:
        print("Generation is not grounded in documents")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("route question")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.data_source == "websearch":
        return WEBSEARCH
    elif source.data_source == "vectorstore":
        return RETRIEVE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_questions,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")