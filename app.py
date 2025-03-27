from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

from agno.agent import Agent
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.models.ollama import Ollama

# Data Retrieval Pipeline
urls = [
    "https://github.com/qdrant/fastembed/",
]

loader = WebBaseLoader(urls)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=50
)

chunks = text_splitter.split_documents(data)
embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")

# Vector Store
client = QdrantClient(path="/tmp/app")
collection_name = "agent-rag"

try:
    collection_info = client.get_collection(collection_name=collection_name)
except (UnexpectedResponse, ValueError):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
) # vector store is created

vector_store.add_documents(documents=chunks) # adding chuinks to vector store
retriever = vector_store.as_retriever() # langchain retriver

knowledge_base = LangChainKnowledgeBase(retriever=retriever) # defining knowledge base using agno

# agno agent framework
agent = Agent(
    model=Ollama(id="llama3.2:3b"),
    knowledge=knowledge_base,
    description="Answer to the user question from the knowledge base",
    markdown=True,
    search_knowledge=True,
)

user_query = "what is FastEmbed?"
agent.print_response(user_query, stream=True)