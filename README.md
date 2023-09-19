# vector_db (windows)

Vector embeddings using serverless NoSQL Astra DB from DataStax.

1. Sign up for an account with DataStax and crate a vector DB.Make sure to save your DB credentials.

2. Install the virtual environment 

# make sure pip is up-to-date and running
py -m pip install --upgrade pip
py -m pip --version

# install virtualenv 
py -m pip install --user virtualenv

# create a virtual environment
py -m venv env

# activate a virtual env. The `(env)` will show in fornt of your command line.
.\env\Scripts\activate

3. Install packages 
pip install cassio datasets langchain openai tiktoken