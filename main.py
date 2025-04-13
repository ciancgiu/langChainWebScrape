
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os 
from langchain_community.document_loaders import SeleniumURLLoader
import shutil




loader = SeleniumURLLoader(urls=["https://www.coursera.org/specializations/deep-learning/paidmedia?utm_medium=sem&utm_source=gg&utm_campaign=b2c_namer_deep-learning_deeplearning-ai_ftcof_specializations_px_dr_bau_gg_sem_pr-bd_us-ca_en_m_hyb_17-08_x&campaignid=904733485&adgroupid=46370300620&device=c&keyword=coursera%20machine%20learning&matchtype=b&network=g&devicemodel=&creativeid=415429098219&assetgroupid=&targetid=aud-543736593054:kwd-297783556067&extensionid=&placement=&gad_source=1&gclid=CjwKCAjwtdi_BhACEiwA97y8BN9PPa_zRuBmWizeFyZEZWv6pwxKVlBDkqmoC2uvEfLQIVcII4bNABoCMyQQAvD_BwE"])
docs = loader.load()

if not docs:
    print("No docs loaded")
else:
    print(docs)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    
)
splits = splitter.split_documents(docs)

if not splits:
    raise ValueError("No documents to add to the vector store.")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="scraped_info",
    embedding_function=embeddings
    )
if add_documents:
    vector_store.add_documents(documents=splits)
    print(f"{len(splits)} added to db")
else:
    print(f"Db already exists")

retriever = vector_store.as_retriever(
    search_type="similarity",
    kwargs={"k":3}
) 


model = OllamaLLM(model ="llama3.2")

template = """

Answer the users questions about the data.

Here is the data we retrieved: {data}

Here is the question to answer: {question}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("What would you like to know about this data? (q to quit)")
    if(question) == "q":
        break
    data = retriever.invoke(question)
    result = chain.invoke({"data":data,"question": question})
    
    print(result)

shutil.rmtree(db_location)