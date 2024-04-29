import requests
from bs4 import BeautifulSoup
import csv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_voyageai import VoyageAIEmbeddings
import wikipediaapi

def generate_embeddings(url):
    wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='Reverse Akinator',
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    p_wiki = wiki_wiki.page(url)
    data = p_wiki.text
    # embeddings = OpenAIEmbeddings()
    embeddings = VoyageAIEmbeddings(
        model="voyage-lite-02-instruct"
    )
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_text(data)
    all_splits = text_splitter.create_documents([data])
    
    db = Chroma.from_documents(
        all_splits, 
        embeddings,
        # persist_directory='./chroma-embeddings',
        collection_name="rag-chroma-voyage"
        # collection_name="rag-chroma-openai"
    )
    return db

# generate_embeddings('Dulquer Salmaan')