from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain import HuggingFaceHub
from langchain.agents import Tool, AgentType
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import GPT4AllEmbeddings
from langchain.agents import initialize_agent
from llama_index import ServiceContext
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from PIL import Image


import nest_asyncio
nest_asyncio.apply()

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llama_index.llms import HuggingFaceInferenceAPI, HuggingFaceLLM
from langchain import HuggingFaceHub
from langchain.agents import Tool, AgentType
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import GPT4AllEmbeddings
from langchain.agents import initialize_agent
from llama_index import ServiceContext
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from transformers import pipeline

from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, VectorStoreIndex
from llama_index.response.pprint_utils import pprint_response

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

HF_TOKEN = "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"
#model_id = "google/flan-t5-large"
model_id = "tiiuae/falcon-7b-instruct"
#embeddings = HuggingFaceHubEmbeddings(repo_id=model_id,huggingfacehub_api_token=HF_TOKEN)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
loader = PyPDFLoader("D:\dev\maintenance_app\SOP_Manuals\Abhimanyu Rai CV.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=100)
texts = text_splitter.split_documents(documents)


docsearch = Chroma.from_documents(texts, embeddings, collection_name="abhimanyu-cv")


# llm = HuggingFaceHub(repo_id = model_id, model_kwargs = {"temperature": 0.1, "max_length": 512},huggingfacehub_api_token=HF_TOKEN)
llm = HuggingFaceHub(repo_id = model_id, model_kwargs = {"temperature": 0.3, "max_length": 1024, "max_new_tokens":2000},huggingfacehub_api_token=HF_TOKEN)
# generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
#                          trust_remote_code=True, device_map="auto", return_full_text=True)

# prompt = PromptTemplate(
#     input_variables=["instruction"],
#     template="{instruction}")
# llm = HuggingFacePipeline(pipeline=generate_text)

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

cv_ask.run("What jobs did Abhimanyu do")

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


# memory = ConversationBufferMemory(memory_key="chat_history")

# agent_executor = initialize_agent(
#     tools, llm, agent="conversational-react-description", memory=memory, verbose=True, handle_parsing_errors="Check your input and try again"
# )



agent_executor = initialize_agent(
     tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors="Check your input and try again", max_iterations=3
 )

agent_executor.run("What jobs did Abhimanyu do")
agent_executor.run("Who is joe biden")



