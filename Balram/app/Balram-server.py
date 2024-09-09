import os
import signal
from typing import List
from fastapi import FastAPI
from langchain_cohere.chat_models import ChatCohere
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
from langchain_cohere.chat_models import ChatCohere
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere.embeddings import CohereEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

load_dotenv()

cohere_api_key = os.environ['COHERE_API_KEY']
tavily_api_key = os.environ['TAVILY_API_KEY']

llm = ChatCohere(cohere_api_key=cohere_api_key, temperature=0.3)

# load retriever
import bs4

loader = WebBaseLoader(web_paths=[
    'https://en.wikipedia.org/wiki/Pest_control#:~:text=In%20agriculture%2C%20pests%20are%20kept,of%20a%20certain%20pest%20species.',
    'https://cpdonline.co.uk/knowledge-base/food-hygiene/pest-control/#the-laws-around-pest-control'],
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=('mw-content-ltr mw-parser-output', 'wpb_wrapper')
                       )))
data = loader.load()
text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)

docs = text_splitter.split_documents(data)


# Embedding

embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key,model='embed-english-v3.0')
vector = FAISS.from_documents(docs, embeddings)
retriever = vector.as_retriever()

base_compressor = CohereRerank()

compression_retriever = ContextualCompressionRetriever(base_compressor=base_compressor, base_retriever=retriever)

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300),
    description='This tool is designed to handle general crop management, pest control, and basic farming queries, \
        including agricultural techniques, common farming challenges, and recommended practices. Use this for queries like how to manage soil fertility,\
        pest identification, crop rotation methods, or general farming terms"'
)
retriever_tool = create_retriever_tool(
    retriever=compression_retriever,
    name='Vectored DB',
    description='This tool specializes in providing detailed information about pesticides, including their types, usage, safety guidelines, \
        and application methods. Use this tool to answer queries related to pest control chemicals, organic pesticides, and how to apply them safely."'
)


tavily = TavilySearchResults(
    max_results=3,
    include_answer=True,
    include_raw_content=True,
    description='This tool is for retrieving real-time information related to agriculture, including weather forecasts, current market prices,\
        and other time-sensitive farming data. Use this tool when the user asks about today’s crop prices, current weather, or anything that requires up-to-date information."'
)


tools = [wikipedia, retriever_tool, tavily]

memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history')

agent_prompt = hub.pull('hwchase17/structured-chat-agent')
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, handle_parsing_errors=True)

app = FastAPI()

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        extra={"widget": {"type": "chat", "input": "location"}},
    )

class Output(BaseModel):
    output: str
        
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

LOCK_FILE = 'server.lock'

def cleanup_lock_file(signum, frame):
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    print("Server stopping...")
    exit(0)

def start_server():
    if not os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, 'w') as f:
            f.write('running')

        signal.signal(signal.SIGINT, cleanup_lock_file)
        signal.signal(signal.SIGTERM, cleanup_lock_file)

        print("Starting server...")
        try:
            import uvicorn
            uvicorn.run(app, host="localhost", port=9000)
        finally:
            cleanup_lock_file(None, None)
    else:
        print("Server is already running.")

if __name__ == "__main__":
    start_server()