import nest_asyncio
nest_asyncio.apply()

import pandas as pd


from llama_index.llms import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding

from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, VectorStoreIndex
from llama_index.response.pprint_utils import pprint_response
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index import set_global_service_context
from llama_index import set_global_tokenizer
from llama_index import LLMPredictor
from llama_index.node_parser import SimpleNodeParser

from transformers import AutoTokenizer

from langchain import HuggingFaceHub
from langchain.llms import GPT4All
import nltk
nltk.download()


HF_TOKEN = "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"
#model_id = "google/flan-t5-large"
model_id = "tiiuae/falcon-7b-instruct"
#embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

set_global_tokenizer(
    AutoTokenizer.from_pretrained(model_id).encode
)

# llm = GPT4All(
#     model="D:\mistral-7b-openorca.Q4_0.gguf", temp=0.1
# )
llm = HuggingFaceInferenceAPI(model_name = model_id, context_window = 1024,token=HF_TOKEN)
#llm = HuggingFaceHub(repo_id = model_id, model_kwargs = {"temperature": 0.3, "max_length": 1024, "max_new_tokens":2000},huggingfacehub_api_token=HF_TOKEN)
llm_predictor = LLMPredictor(llm=llm)

service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
set_global_service_context(service_context=service_context)


sop_documents = SimpleDirectoryReader("D:\dev\maintenance_app\SOP_Manuals").load_data()
maintenace_records = SimpleDirectoryReader("D:\dev\maintenance_app\Maintenance_Records").load_data()

parser = SimpleNodeParser.from_defaults(chunk_size=200, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(sop_documents)


sop_index = VectorStoreIndex(nodes=nodes,service_context=service_context)
query_engine = sop_index.as_query_engine()
response = query_engine.query("Give me 3 of Abhimanyu's top qualifications")
print(response)


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



