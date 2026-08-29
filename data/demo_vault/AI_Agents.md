# AI Agents & Autonomous Reasoning

#ai/agents #reasoning #llm

## What is an AI Agent?
An **AI Agent** is an autonomous system powered by a Large Language Model ([[LLMs]]) that perceives its environment, reasons through a sequence of steps, makes decisions, and invokes external tools or APIs to accomplish a specified objective.

While standard [[RAG]] is a one-shot retrieve-and-generate pipeline, an Agent can iteratively search, evaluate retrieved information, refine its search queries, and interact with multiple systems.

## Key Components of Agent Architecture
1. **Core LLM (The Brain)**: Handles natural language understanding, task decomposition, and logical reasoning.
2. **Planning & Reflection**:
   - Deconstructs complex goals into manageable subtasks.
   - Self-reflects on past actions to correct mistakes (e.g., Reflexion framework).
3. **Memory**:
   - **Short-term Memory**: In-context conversational history and scratchpad reasoning.
   - **Long-term Memory**: External vector databases ([[Vector_Databases]]) storing past interactions or domain documents for persistent retrieval.
4. **Tools & Actuators**:
   - Functions the agent can execute, such as web searching, code execution, database lookups, or file modifications.

## Agent Reasoning Patterns
### The ReAct Pattern (Reason + Act)
The ReAct framework alternates between reasoning steps and action steps:
- **Thought**: The model plans what to do next based on the user's objective and previous observations.
- **Action**: The model chooses a specific tool and inputs arguments (e.g., `search_vault(query="ChromaDB HNSW")`).
- **Observation**: The system executes the action and feeds the output back into the prompt.
- **Final Answer**: Once sufficient information is gathered, the agent generates the final response.

## Difference Between RAG and Agents
- **Standard RAG**: User Question $\rightarrow$ Single Retrieval Step $\rightarrow$ Single Generation Step $\rightarrow$ Output.
- **Agentic RAG**: User Question $\rightarrow$ Query Reformulation $\rightarrow$ Retrieval $\rightarrow$ Evaluation of Relevance $\rightarrow$ Optional Secondary Retrieval $\rightarrow$ Synthesis.

## Related Concepts
- [[LLMs]]
- [[RAG]]
- [[Vector_Databases]]
