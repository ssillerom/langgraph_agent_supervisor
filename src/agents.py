from typing import Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, END
from langgraph.types import Command

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


MEMBERS = ["investigador", "programador"]

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
OPTIONS = MEMBERS + ["FINALIZAR"]

system_prompt = (
    "Eres un supervisor encargado de gestionar una conversación entre los"
    f" siguientes trabajadores: {MEMBERS}. Dada la siguiente solicitud del usuario,"
    " responde con el trabajador que debe actuar a continuación. Cada trabajador realizará una"
    " tarea y responderá con sus resultados y estado. Si la tarea involucra operaciones matemáticas,"
    " asegúrate de NO asignarla al investigador, sino al programador. Cuando haya finalizado,"
    " responde con FINALIZAR."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINALIZAR."""

    next: Literal[*OPTIONS]


class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*MEMBERS, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = LLM.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINALIZAR":
        goto = END

    return Command(goto=goto, update={"next": goto})