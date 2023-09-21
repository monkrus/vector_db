import json
from datasets import load_dataset
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores.cassandra import Cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# Vector support using Langchain, Apache Cassandra (Astra DB is built using
# Cassandra), and OpenAI (to generate embeddings)

# Constants
ASTRA_DB_SECURE_BUNDLE_PATH = "YUR_ASTRADB_SECURE_BUNDLE_PATH"
ASTRA_DB_TOKEN_JSON_PATH = "YOUR_ASTRADB_TOKEN_JSON_PATH"
ASTRA_DB_KEYSPACE = "YOUR_ASTRADB_KEYSPACE"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"


# Load Astra DB application token from a JSON file
with open(ASTRA_DB_TOKEN_JSON_PATH) as f:
    secrets = json.load(f)
ASTRA_DB_APPLICATION_TOKEN = secrets["token"]

# Connect to Astra DB
cloud_config = {"secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH}
auth_provider = PlainTextAuthProvider("token", ASTRA_DB_APPLICATION_TOKEN)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astra_session = cluster.connect()

# Initialize OpenAI and embeddings
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
my_embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create Cassandra vector store
my_cassandra_vstore = Cassandra(
    embedding=my_embedding,
    session=astra_session,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="qa_mini_demo"
)

# Load data from Hugging Face
print("Loading data from Hugging Face")
my_dataset = load_dataset("Biddls/Onion_News", split="train")
headlines = my_dataset["text"][:50]

# Generate embeddings and store in AstraDB
print("\nGenerating embeddings and storing in AstraDB")
my_cassandra_vstore.add_texts(headlines)

print("Inserted %i headlines.\n" % len(headlines))

# Initialize vector index
vector_index = VectorStoreIndexWrapper(vectorstore=my_cassandra_vstore)

# Interactive question answering loop
first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ")
        first_question = False
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ")

    if query_text.lower() == "quit":
        break

    print("QUESTION: \"%s\"" % query_text)
    answer = vector_index.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in my_cassandra_vstore.similarity_search_with_score(query_text, k=4):
        print("  %0.4f \"%s ...\"" % (score, doc.page_content[:60]))
