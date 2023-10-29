# Code Talk
Code Talk is a library that leverages the Python binding of Meta's Llama model to help users 'talk' to their code. This is achieved through in context learning through which the user can provide the model the codebase that they would like to interact with and through the usage of a vector store and a similarity search, given a query the model learns new information from the provided documents dynamically.

## Prerequisites 
Install langchain 

```
pip install langchain
```  

Install FAISS vector-store
```
pip install faiss-cpu
``` 

Install llama-cpp-python:

llama-cpp-python can be installed to either run on CPU or GPU 
    
 To install llama-cpp-python to run on CPU:
 ```
 pip install llama-cpp-python
 ``` 
    
 To install llama-cpp-python to run on GPU:
 Set the CMAKE_ARGS and FORCE_CMAKE environment variables
 To install llama-cpp-python with cuBLAS:
 ```
 CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
 FORCE_CMAKE = 1
 ```
 To install llama-cpp-python with OpenBLAS: 
 ```
 CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
 FORCE_CMAKE=1
 ```
 To install llama-cpp-python CLBlast:
 ```
 CMAKE_ARGS="-DLLAMA_CLBLAST=on"
 FORCE_CMAKE=1
 ```
 Finally, install llama-cpp-python
 ```
 pip install llama-cpp-python
 ```
Install hugging-faces 
```
pip install huggingface-hub
```
## Installing
Clone the repo:
```
git clone https://github.com/jayanthd04/Code-Talk.git
```

Delete '''Code-Talk/documents/llama-cpp-python''' and add the codebase that you would like to interact with to '''Code-Talk/documents''' folder   

Run codeTalk.py

After the first run of the code, the following segments of code can be commented out since we do not need to chunk the files and embed them each time.

```
root_dir = './documents/llama-cpp-python'
docs= []
for dirpath,dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader=TextLoader(os.path.join(dirpath,file),encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts=text_splitter.split_documents(docs)
``` 

```
faiss_index = FAISS.from_documents(texts,embeddings)
faiss_index.save_local("./embeddings/codetalk")
```

Here's an example of the model running :
```
Enter a question:
 How do I use the high-level API to run a Llama-cpp model on GPU?
Llama.generate: prefix-match hit

llama_print_timings:        load time =   739.73 ms
llama_print_timings:      sample time =     4.77 ms /    22 runs   (    0.22 ms per token,  4609.26 tokens per second)
llama_print_timings: prompt eval time = 50267.16 ms /   876 tokens (   57.38 ms per token,    17.43 tokens per second)
llama_print_timings:        eval time =  1608.41 ms /    21 runs   (   76.59 ms per token,    13.06 tokens per second)
llama_print_timings:       total time = 52106.46 ms
Llama.generate: prefix-match hit

llama_print_timings:        load time =   739.73 ms
llama_print_timings:      sample time =    17.89 ms /    86 runs   (    0.21 ms per token,  4806.62 tokens per second)
llama_print_timings: prompt eval time = 89284.10 ms /  1398 tokens (   63.87 ms per token,    15.66 tokens per second)
llama_print_timings:        eval time =  7529.02 ms /    85 runs   (   88.58 ms per token,    11.29 tokens per second)
llama_print_timings:       total time = 97316.52 ms
 You can use the high-level API to run a Llama-cpp model on GPU by setting the `n_gpu_layers` parameter when creating the FastAPI application. For example, you can set `n_gpu_layers=1` to use one GPU layer for the model. Additionally, you can specify the path to your ggml model file using the `MODEL` environment variable.
```
**Note:** The above example was run with the verbose flag set to be true, if you do not want to view the model run stats you can set verbose to be false in the following code segment in the codeTalk.py file 

```
llm=LlamaCpp(model_path=llm_path,n_ctx=2000,callback_manager=callback_manager,verbose=True,use_mlock=True,n_gpu_layers=30,n_threads=4,max_tokens=4000)
```
