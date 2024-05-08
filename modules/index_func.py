import hashlib
import logging
import os
import boto3
import PyPDF2
from typing import List, Union
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
from langchain.embeddings import BedrockEmbeddings
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from modules.config import local_embedding
from modules.presets import *
from modules.utils import *

# 配置参数
MAX_INPUT_SIZE = 4096
NUM_OUTPUTS = 5
MAX_CHUNK_OVERLAP = 50
CHUNK_SIZE_LIMIT = 3000
EMBEDDING_LIMIT = None
SEPARATOR = " "
LOAD_FROM_CACHE_IF_POSSIBLE = True

# 支持的文件类型
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".pptx", ".epub", ".xlsx"]

def load_document(filepath: str) -> List[Document]:
    """
    加载指定文件路径的文档,并按指定的分块大小和重叠长度进行分块
    :param filepath: 文件路径
    :return: 分块后的文档列表
    """
    filename = os.path.basename(filepath)
    file_type = os.path.splitext(filename)[1]
    logging.info(f"Loading file: {filename}")

    if file_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type: {file_type}")

    try:
        if file_type == ".pdf":
            logging.debug("Loading PDF...")
            pdftext = ""
            with open(filepath, "rb") as pdfFileObj:
                pdfReader = PyPDF2.PdfReader(pdfFileObj)
                for page in tqdm(pdfReader.pages):
                    pdftext += page.extract_text()
            texts = [Document(page_content=pdftext, metadata={"source": filepath})]
        elif file_type == ".docx":
            logging.debug("Loading Word...")
            from langchain.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(filepath)
            texts = loader.load()
        elif file_type == ".pptx":
            logging.debug("Loading PowerPoint...")
            from langchain.document_loaders import UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(filepath)
            texts = loader.load()
        elif file_type == ".epub":
            logging.debug("Loading EPUB...")
            from langchain.document_loaders import UnstructuredEPubLoader
            loader = UnstructuredEPubLoader(filepath)
            texts = loader.load()
        elif file_type == ".xlsx":
            logging.debug("Loading Excel...")
            text_list = excel_to_string(filepath)
            texts = [Document(page_content=text, metadata={"source": filepath}) for text in text_list]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logging.error(f"Error loading file: {filename}", exc_info=True)
        raise e

    text_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE_LIMIT, chunk_overlap=MAX_CHUNK_OVERLAP)
    texts = text_splitter.split_documents(texts)
    return texts

def get_documents(file_src: List[Union[str, os.PathLike]]) -> List[Document]:
    """
    从指定的文件列表中加载所有文档
    :param file_src: 文件路径列表
    :return: 所有文档的列表
    """
    documents = []
    logging.debug("Loading documents...")
    for file in file_src:
        try:
            documents.extend(load_document(str(file)))
        except Exception as e:
            logging.error(f"Error loading file: {file}", exc_info=True)
    logging.debug("Documents loaded.")
    return documents

def construct_index(
    api_key: str,
    file_src: List[Union[str, os.PathLike]],
    max_input_size: int = MAX_INPUT_SIZE,
    num_outputs: int = NUM_OUTPUTS,
    max_chunk_overlap: int = MAX_CHUNK_OVERLAP,
    chunk_size_limit: int = CHUNK_SIZE_LIMIT,
    embedding_limit: int = EMBEDDING_LIMIT,
    separator: str = SEPARATOR,
    load_from_cache_if_possible: bool = LOAD_FROM_CACHE_IF_POSSIBLE,
) -> FAISS:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        # 由于一个依赖的愚蠢的设计，这里必须要有一个API KEY
        os.environ["OPENAI_API_KEY"] = "sk-xxxxxxx"
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    logging.debug(f"API base: {os.environ.get('OPENAI_API_BASE', None)}")
    chunk_size_limit = None if chunk_size_limit == 0 else chunk_size_limit
    embedding_limit = None if embedding_limit == 0 else embedding_limit
    separator = " " if separator == "" else separator

    index_name = get_file_hash(file_src)
    index_path = f"./index/{index_name}"
    if local_embedding:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
    else:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

    if os.path.exists(index_path) and load_from_cache_if_possible:
        logging.info("Found cached index, loading...")
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error("Error loading cached index", exc_info=True)
    else:
        documents = get_documents(file_src)
        logging.debug("Building index...")
        if documents:
            with retrieve_proxy():
                try:
                    index = FAISS.from_documents(documents, embeddings)
                except Exception as e:
                    logging.error("Error building index", exc_info=True)
                    raise e
        else:
            raise Exception("No supported documents found.")
        logging.debug("Index built successfully!")
        os.makedirs("./index", exist_ok=True)
        try:
            index.save_local(index_path)
        except Exception as e:
            logging.error("Error saving index", exc_info=True)
            raise e
        logging.debug("Index saved locally!")
        return index