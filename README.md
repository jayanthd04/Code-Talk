# Code Talk
Code Talk is a library that leverages the Python binding of Meta's Llama model to help users 'talk' to their code. This is achieved through in context learning through which the user can provide the model the codebase that they would like to interact with and through the usage of a vector store and a similarity search, given a query the model learns new information from the provided documents dynamically.

## Installing required packages 
Install langchain 
'''
pip install langchain
'''  

Install FAISS vector-store
'''
pip install faiss-cpu
''' 

Install llama-cpp-python:

    llama-cpp-python can be installed to either run on CPU or GPU 
    
    To install llama-cpp-python to run on CPU:
    '''
    pip install llama-cpp-python
    ''' 
    
    To install llama-cpp-python to run on GPU:
        Set the CMAKE_ARGS and FORCE_CMAKE environment variables
        To install llama-cpp-python with cuBLAS:
            '''
            CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
            FORCE_CMAKE = 1
            '''
        To install llama-cpp-python with OpenBLAS: 
            '''
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
            FORCE_CMAKE=1
            '''
        To install llama-cpp-python CLBlast:
            '''
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
            FORCE_CMAKE=1
            '''
        Finally, install llama-cpp-python
        '''
        pip install llama-cpp-python
        '''
Install hugging-faces 
'''
pip install huggingface-hub
'''
