import nest_asyncio
nest_asyncio.apply()

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llama_index.llms import HuggingFaceInferenceAPI, HuggingFaceLLM
from langchain import HuggingFaceHub
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import GPT4AllEmbeddings
from langchain.agents import initialize_agent
from llama_index import ServiceContext
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA

from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, VectorStoreIndex
from llama_index.response.pprint_utils import pprint_response

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


loader = PyPDFLoader("D:\dev\maintenance_app\SOP_Manuals\Abhimanyu Rai CV.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

docsearch = Chroma.from_documents(texts, embeddings, collection_name="abhimanyu-cv")

llm = GPT4All(
    model="D:\mistral-7b-openorca.Q4_0.gguf", temp=0.1
)


# sop_documents = SimpleDirectoryReader("./SOP_Manuals").load_data()
# maintenace_records = SimpleDirectoryReader("./Maintenance_Records").load_data()

# service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

# sop_index = VectorStoreIndex.from_documents(documents=sop_documents,service_context=service_context)



# index = VectorStoreIndex.from_documents(documents=documents,service_context=service_context)

# tools = [
#     Tool(
#         name="LlamaIndex",
#         func=lambda q: str(index.as_query_engine().query(q)),
#         description="useful for when you want to answer questions about Abhimanyu. The input to this tool should be a complete english sentence.",
#         return_direct=True,
#     ),
# ]


cv_ask = RetrievalQA.from_chain_type(

    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

cv_ask.run("What are Abhimanyu's extracurricular activities")
prompt = PromptTemplate.from_template(
    "{query}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    Tool(
        name="LlamaIndex",
        func=lambda q: str(cv_ask.run(q)),
        description="useful for when you want to answer questions about Abhimanyu",
        return_direct=True,
    ),
    Tool(
        name="LLM",
        func=lambda q: str(llm_chain.run(query = q)),
        description="useful for general purpose queries and logic",
        return_direct=True,
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executor = initialize_agent(
    tools, llm, agent="conversational-react-description", memory=memory, verbose=True
)
agent_executor.run(input="Tell me about Abhimanyu's extracurricular activities")
agent_executor.run("Who is joe biden")

