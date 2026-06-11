import os
from typing import Annotated
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

FREE_MODEL="Qwen/Qwen2.5-7B-Instruct"

def build_client() -> tuple[InferenceClient, str]:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set.\n"
            "  Windows : set HF_TOKEN=hf_...\n"
            "  Linux   : export HF_TOKEN=hf_...\n"
            "  Or add it to a .env file."
        )
    client = InferenceClient(token=hf_token)
    return client, FREE_MODEL


def lc_messages_to_hf(messages: list) -> list[dict]:
    """Convert LangChain message objects → HF chat dicts."""
    role_map = {
        HumanMessage: "user",
        AIMessage:    "assistant",
        SystemMessage: "system",
    }
    result = []
    for m in messages:
        role = role_map.get(type(m), "user")
        content = m.content if hasattr(m, "content") else str(m)
        result.append({"role": role, "content": content})
    return result

def build_graph(client: InferenceClient, model_id: str):

    def chatbot_node(state: State) -> State:
        hf_msgs = lc_messages_to_hf(state["messages"])

        response = client.chat_completion(
            messages=hf_msgs,
            model=model_id,
            max_tokens=512,
            temperature=0.7,
        )
        reply_text = response.choices[0].message.content
        return {"messages": [AIMessage(content=reply_text)]}

    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

def run():
    print("Connecting to HuggingFace Inference API …")
    client, model_id = build_client()
    graph = build_graph(client, model_id)

    config = {"configurable": {"thread_id": "session-1"}}

    print(f"║  LangGraph × HuggingFace Chatbot        ║")
    print(f"║  Model : {model_id[:34]:<34}║")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        # Last message in state is the assistant reply
        reply = result["messages"][-1].content
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    run()