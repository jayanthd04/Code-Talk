import os 
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

"""root_dir = './documents/llama-cpp-python'
docs= []
for dirpath,dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader=TextLoader(os.path.join(dirpath,file),encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts=text_splitter.split_documents(docs)"""
embeddings= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
"""faiss_index = FAISS.from_documents(texts,embeddings)
faiss_index.save_local("./embeddings/codetalk")"""
print("loading indexes")
faiss_index=FAISS.load_local("./embeddings/codetalk",embeddings)
retriever = faiss_index.as_retriever()
retriever.search_kwargs['distance_metric']='cos'
retriever.search_kwargs['fetch_k']=100
retriever.search_kwargs['maximal_marginal_relevance']=True
print("index loaded")
llm_path = './models/llama-2-7b-chat.ggmlv3.q4_1.bin'
callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler])
llm=LlamaCpp(model_path=llm_path,n_ctx=2000,callback_manager=callback_manager,verbose=True,use_mlock=True,n_gpu_layers=30,n_threads=4,max_tokens=4000)
qa = ConversationalRetrievalChain.from_llm(llm,retriever=retriever)
chat_history = []
while(True):
    print("Enter a question: ")
    question=input()
    result = qa({"question":question,"chat_history":chat_history})
    chat_history.append((question,result['answer']))
    print(result['answer'])

