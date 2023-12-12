import pandas as pd


from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.embeddings import GPT4AllEmbeddings
from langchain.agents import initialize_agent
from llama_index import ServiceContext
from llama_index.indices.struct_store import GPTPandasIndex
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from transformers import pipeline
from PIL import Image

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
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine.pandas_query_engine import PandasQueryEngine
from llama_index.response.pprint_utils import pprint_response
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index import StorageContext, load_index_from_storage
import os
import openai
from llama_index.vector_stores import ChromaVectorStore
import streamlit as st
import streamlit_authenticator as sa

#app_path = "D:/dev/maintenance_app"
#app_path = "."
logo_path = app_path+"/Files/logo.png"


os.environ["OPENAI_API_KEY"] = st.secrets["open_ai"]

openai.api_key = os.environ["OPENAI_API_KEY"]


st.set_page_config(page_title="Maintenance Bot", page_icon=logo_path)
cola,colb = st.columns([5,1])
with cola:
    st.title("Welcome to the Maintenance Bot for Mining")
    st.write("Provides all information regarding a given maintenance record, equipment details and provides recommendations on maintenance best practices")
with colb:
    img = Image.open(logo_path)

    st.image(img, width=150)  
st.markdown('---')



#df= pd.read_excel(app_path+"/Maintenance_Records/Maintenance Log.xlsx")

# db = chromadb.PersistentClient(path="./chroma_db")
# overall_collection = db.get_or_create_collection("Plant_Maintenance_Manual")
# maintenance_collection = db.get_or_create_collection("Caterpillar_Maintenance_Manual")




# def create_chroma_store_pdf(path,_collection):
#     reader = SimpleDirectoryReader(input_files=[path]).load_data()
#     parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=100)
#     nodes = parser.get_nodes_from_documents(reader)
    
   
#     vector_store = ChromaVectorStore(chroma_collection=_collection)
    
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     index = VectorStoreIndex(nodes=nodes,storage_context=storage_context)
#     engine = index.as_query_engine(similarity_top_k=3)
#     return engine
@st.cache_data
def create_chroma_store_pdf(path):
    
    
    try: 
        storage_context = StorageContext.from_defaults(persist_dir=path)
    except FileNotFoundError as f:
        print("Storage context is empty.")
        print(path)
        reader = SimpleDirectoryReader(path).load_data()
        parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=100)
        nodes = parser.get_nodes_from_documents(reader)
        storage_context=StorageContext.from_defaults()
        index = VectorStoreIndex(nodes=nodes)
        index.storage_context.persist(persist_dir=path)
        
    else:    
        print("Storage context contains data.")
        print(path)
        index = load_index_from_storage(storage_context)
    
    
    engine = index.as_query_engine(similarity_top_k=3)
    return engine

# @st.cache_data
# def update_data_vector(path,df):
    
    
#     try: 
#         storage_context = StorageContext.from_defaults(persist_dir=path)
#     except FileNotFoundError as f:
#         print("Storage context is empty.")
#         print(path)
#         reader = SimpleDirectoryReader(path).load_data()
#         # need to convert reader to df

#         index = GPTPandasIndex (df=df)
#         #need to change df here to the existing reader value
#         index.storage_context.persist(persist_dir=path)
        
#     else:    
#         print("Storage context contains data.")
#         print(path)
#         index = load_index_from_storage(storage_context)
    
    
#     engine = index.as_query_engine()
#     return engine



# data_vector_index = update_data_vector(path = app_path+"/Maintenance_Records")
caterpillar_maintenance_engine = create_chroma_store_pdf(path = app_path+"/SAP_Manuals_Caterpillar")
caterpillar_specs_engine = create_chroma_store_pdf(path = app_path+"/Specs")

overall_maintenance_engine = create_chroma_store_pdf(path = app_path+"/SOP_Manuals")


# overall_maintenance_index.storage_context.persist(persist_dir=app_path+"/Index")
# caterpillar_maintenance_index.storage_context.persist(persist_dir=app_path+"/Index")

# overall_maintenance_engine = overall_maintenance_index.as_query_engine(similarity_top_k=3)
# caterpillar_maintenance_engine = caterpillar_maintenance_index.as_query_engine(similarity_top_k=3)

# overall_maintenance = SimpleDirectoryReader(input_files=[app_path+"/SOP_Manuals/Maintenace_Manual_Overall.pdf"]).load_data()
# caterpillar_maintenance = SimpleDirectoryReader(input_files=[app_path+"/SOP_Manuals/Maintenance_Manual_Caterpillar_785_789_793.pdf"]).load_data()

# parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=100)

# overall_maintenance_nodes = parser.get_nodes_from_documents(overall_maintenance)
# caterpillar_maintenance_nodes = parser.get_nodes_from_documents(caterpillar_maintenance)




# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection_overall_maintenance = db.get_or_create_collection("Plant_Maintenance_Manual")
# chroma_collection_cat_maintenance = db.get_or_create_collection("Caterpillar_Maintenance_Manual")
# vector_store_overall_maintenance = ChromaVectorStore(chroma_collection=chroma_collection_overall_maintenance)
# vector_store_cat_maintenance = ChromaVectorStore(chroma_collection=chroma_collection_cat_maintenance)
# storage_context_overall_maintenance = StorageContext.from_defaults(vector_store=vector_store_overall_maintenance)
# storage_context_cat_maintenance = StorageContext.from_defaults(vector_store=vector_store_cat_maintenance)
# overall_maintenance_index = VectorStoreIndex(nodes=overall_maintenance_nodes,storage_context=storage_context_overall_maintenance)
# caterpillar_maintenance_index = VectorStoreIndex(nodes=caterpillar_maintenance_nodes,storage_context=storage_context_cat_maintenance)

# # overall_maintenance_index.storage_context.persist(persist_dir=app_path+"/Index")
# # caterpillar_maintenance_index.storage_context.persist(persist_dir=app_path+"/Index")

# overall_maintenance_engine = overall_maintenance_index.as_query_engine(similarity_top_k=3)
# caterpillar_maintenance_engine = caterpillar_maintenance_index.as_query_engine(similarity_top_k=3)






##### Streamlit #####




  




openai_api_key = st.secrets["open_ai"]

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Only Excel Files are supported",
   
)


with st.sidebar:
    img = Image.open(logo_path)

    # st.sidebar.image(img, width=70, use_column_width=True)  
    

    st.title('Maintenance Bot for Mining')
    st.subheader('Powered by Accenture')

    hf_pass = st.text_input('Enter password:', type='password')
    if not (hf_pass == st.secrets['password']):
        st.warning('Please enter your credentials!', icon='⚠️')
        
    else:
        st.success('Proceed to entering your query!', icon='👉')


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None
    
st.markdown('---')

if uploaded_file:
    df = load_data(uploaded_file)
    
    pandas_query_engine = PandasQueryEngine(df=df,verbose=True)



    query_engine_tools = [
        QueryEngineTool(
            query_engine=overall_maintenance_engine,
            metadata=ToolMetadata(name='Plant Maintenance Manual', description='Provides summary information about Maintenance Manual practices for all equipment in a mine')
        ),
        QueryEngineTool(
            query_engine=caterpillar_maintenance_engine,
            metadata=ToolMetadata(name='Cat 785 Cat 789 Cat 793 Maintenance Manual', description='Provides information about maintenance & service practices for Caterpillar - Cat  785 Cat 789 Cat 793 series trucks')
        ),
        QueryEngineTool(
            query_engine=pandas_query_engine,
            metadata=ToolMetadata(name='Maintenance Log', description='Provides base data of equipment health and maintenance including breakdown for a given mine')
        ),
        QueryEngineTool(
            query_engine=caterpillar_specs_engine,
            metadata=ToolMetadata(name='Specs for Caterpilar', description='Provides infromation about caterpillar equipment specifications')
        ),
    ]

    #panda_index = GPTPandasIndex (df=df)
    #pandas_query_engine = panda_index.as_query_engine()
    llm = OpenAI(temperature=0,model="gpt-3.5-turbo")

    # Given a query, this query engine `SubQuestionQueryEngine ` will generate a “query plan”
    # containing sub-queries against sub-documents before synthesizing the final answer.
    s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)


    # response = s_engine.query("Which machine shows the highest equipment breakdown? Give me 2-3 recommendations for its maintenance")

    # response.response


    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "Maintenance Bot", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Please post your query"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    if prompt is not None:
        with st.chat_message("Maintenance Bot"):
                response = s_engine.query(prompt)
                st.session_state.messages.append({"role": "Maintenance Bot", "content": response.response})
                st.write(response.response)

