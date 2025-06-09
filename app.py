import gradio as gr
from os import getenv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import Tool
from typing import TypedDict, Annotated

# RAG IMPORTS
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# LANGGRAPH IMPORTS (para usar tools)
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# region RAG
md_path = "./menu.md"  # Path to your markdown file

with open(md_path, "r", encoding="utf-8") as file:
    md_content = file.read()

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Titulo"),
        ("##", "Plato")
    ], 
    strip_headers=False)
splits = splitter.split_text(md_content)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore.from_documents(splits, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant documents
# endregion

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="google/gemini-2.5-flash-preview",
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer {getenv('HELICONE_API_KEY')}"
        }
    },
)

@tool
def reservar_mesa(hora: str) -> str:
    """
    Asigna automáticamente una mesa disponible para la hora solicitada.
    Si no hay mesas disponibles, informa al usuario.
    
    Args:
        hora: Hora en formato "HH:MM" (ej: "13:00", "21:00")
    """
    horas_disponibles = ["13:00", "14:00", "15:00", "20:00", "21:00", "22:00", "23:00"]
    mesas_ocupadas = [("13:00", 1), ("13:00", 2), ("13:00", 3), ("13:00", 4), ("13:00", 5), ("13:00", 6), ("14:00", 2), ("14:00", 4), ("15:00", 6),
                      ("20:00", 1), ("20:00", 2), ("20:00", 3), ("21:00", 4), ("21:00", 5), ("22:00", 1), ("22:00", 6)]
    
    # Verificar si el restaurante está abierto a esa hora
    if hora not in horas_disponibles:
        return f"Lo lamento, el restaurante solo está abierto en los siguientes horarios: {', '.join(horas_disponibles)}. Por favor elige una de estas horas."
    
    # Encontrar mesas disponibles para esa hora
    mesas_disponibles = []
    total_mesas = 6  # Tenemos mesas del 1 al 6
    
    for mesa_num in range(1, total_mesas + 1):
        if (hora, mesa_num) not in mesas_ocupadas:
            mesas_disponibles.append(mesa_num)
    
    # Si no hay mesas disponibles
    if not mesas_disponibles:
        return f"Lo siento mucho, no tenemos mesas disponibles a las {hora}. Todas nuestras {total_mesas} mesas están ocupadas en ese horario. ¿Te gustaría probar con otra hora?"
    
    # Asignar la primera mesa disponible
    mesa_asignada = mesas_disponibles[0]
    mesas_restantes = len(mesas_disponibles) - 1
    
    # Simular la reserva (en un caso real, aquí actualizarías la base de datos)
    # mesas_ocupadas.append((hora, mesa_asignada))  # Esto sería en una DB real
    
    if mesas_restantes > 0:
        return f"¡Perfecto! Te he asignado la mesa {mesa_asignada} para las {hora}. Quedan {mesas_restantes} mesas disponibles en ese horario."
    else:
        return f"¡Perfecto! Te he asignado la mesa {mesa_asignada} para las {hora}. Era la última mesa disponible en ese horario."

# Crear el agente con herramientas y memoria
memory = MemorySaver()
agent_executor = create_react_agent(
    llm,
    tools=[reservar_mesa],  # Nueva herramienta de reserva inteligente
    checkpointer=memory
)

def respond(message, history):
    """
    Process user message and generate a streaming response with tools.
    
    Args:
        message: User's input message
        history: List of message dictionaries from Gradio
    
    Returns:
        Generator that yields string responses
    """
    
    # Sección RAG: recoger documentos
    documents = retriever.invoke(message)
    context = "\n\n".join([doc.page_content for doc in documents])
    
    SYSTEM_MESSAGE = f"""
    Eres un camarero experto en un restaurante, tienes una carta de menú con platos y bebidas a tu disposición,
    si el usuario te pregunta por un plato or bebida, debes verificar que existe en la carta.
    
    Para las reservas de mesa:
    - Cuando un cliente quiera reservar una mesa, solo necesitas saber la HORA
    - NO le preguntes qué mesa quiere, tú le asignarás automáticamente la mejor disponible
    - Usa la herramienta reservar_mesa con la hora que el cliente solicite
    - La herramienta te dirá si hay mesas disponibles y cuál se le asignó
    - Si no hay mesas disponibles, ofrece horarios alternativos
    
    INFORMACIÓN DE LA CARTA:
    {context}
    """
    
    # Convertir historial de Gradio a formato LangChain
    messages = [SystemMessage(content=SYSTEM_MESSAGE)]
    
    # Convertir historial (formato tuples) a mensajes de LangChain
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=user_msg))
        if ai_msg:  # Solo añadir si no es None
            messages.append(AIMessage(content=ai_msg))
    
    # Añadir mensaje actual del usuario
    messages.append(HumanMessage(content=message))
    
    # Configuración para la memoria del agente
    config = {"configurable": {"thread_id": "restaurant-chat"}}
    
    # Variable para construir la respuesta completa
    full_response = ""
    tool_used = False
    
    try:
        for chunk in agent_executor.stream({"messages": messages}, config=config):
            # Si el agente usa herramientas
            if "tools" in chunk:
                tool_used = True
                # Mostrar indicador de que se está procesando la reserva
                loading_msg = "🍽️ **Verificando disponibilidad de mesa...**\n\n"
                full_response += loading_msg
                yield full_response
            
            # Si hay respuesta del agente
            if "agent" in chunk:
                for agent_msg in chunk["agent"]["messages"]:
                    # Si se usó una herramienta, limpiar el mensaje de carga
                    if tool_used:
                        full_response = ""  # Limpiar el mensaje de carga
                        tool_used = False
                    
                    full_response += agent_msg.content
                    yield full_response
                    
    except Exception as e:
        # Fallback en caso de error
        full_response = f"Lo siento, hubo un error procesando tu solicitud: {str(e)}"
        yield full_response

# Create and launch the Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title="Restaurante Rafa - Camarero Virtual",
    description="Pregúntame sobre nuestra carta o haz una reserva de mesa",
    examples=[
        "¿Qué platos tenéis?",
        "Quiero reservar una mesa para las 21:00",
        "¿Podéis reservarme para las 14:00?",
        "¿Qué bebidas tenéis?",
        "Mesa para las 22:00 por favor"
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()  