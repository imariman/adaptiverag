# Adaptive RAG

Adaptive RAG is a strategy for RAG that unites (1) [query analysis](https://blog.langchain.dev/query-construction/) 
with (2) [active / self-corrective RAG](https://blog.langchain.dev/agentic-rag-with-langgraph/).

In the paper, they report query analysis to route across:

No Retrieval
Single-shot RAG
Iterative RAG
Let's build on this using LangGraph.

In our implementation, we will route between:

Web search: for questions related to recent events

Self-corrective RAG: for questions related to our index

![Schema](schema.png)
