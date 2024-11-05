import os
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.metrics import precision_score, recall_score, f1_score

# Define a function to load documents from a directory
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()
                    documents.append({"filename": filename, "content": content})
            except Exception as e:
                st.error(f"Error loading file {filepath}: {e}")
    return documents

# Create embeddings using Sentence Transformers
def create_embeddings(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc['content'] for doc in documents]
    embeddings = model.encode(texts)
    return embeddings

# Answer query by finding the top K most relevant documents using a specific similarity measure
def answer_query(documents, embeddings, query, top_k=5, similarity_measure="cosine"):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed the query
    query_embedding = model.encode([query])

    # Choose the similarity measure
    if similarity_measure == "cosine":
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort in descending order
    elif similarity_measure == "euclidean":
        distances = euclidean_distances(query_embedding, embeddings)[0]
        top_indices = np.argsort(distances)[:top_k]  # Sort in ascending order (smaller distances are better)
        similarities = 1 / (1 + distances)  # Invert distances to resemble similarity scores
    elif similarity_measure == "manhattan":
        distances = manhattan_distances(query_embedding, embeddings)[0]
        top_indices = np.argsort(distances)[:top_k]  # Sort in ascending order
        similarities = 1 / (1 + distances)  # Invert distances for similarity-like interpretation
    else:
        raise ValueError(f"Unknown similarity measure: {similarity_measure}")

    # Gather the top K most relevant documents and their similarity scores
    top_documents = [(documents[i]['filename'], documents[i]['content'], similarities[i]) for i in top_indices]

    return top_documents

# Dummy function to get ground truth for a query (in real scenario, you'd need actual labels)
def get_ground_truth(query):
    # Placeholder for the ground truth (list of indices of relevant documents)
    # manually labeled data.
    relevant_docs = [0, 3, 4] 
    return relevant_docs

# Evaluate Precision, Recall, F1-Score
def evaluate_model(documents, embeddings, query, top_k=5, similarity_measure="cosine"):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed the query
    query_embedding = model.encode([query])

    # Choose the similarity measure
    if similarity_measure == "cosine":
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort in descending order
    elif similarity_measure == "euclidean":
        distances = euclidean_distances(query_embedding, embeddings)[0]
        top_indices = np.argsort(distances)[:top_k]  # Sort in ascending order (smaller distances are better)
    elif similarity_measure == "manhattan":
        distances = manhattan_distances(query_embedding, embeddings)[0]
        top_indices = np.argsort(distances)[:top_k]  # Sort in ascending order
    else:
        raise ValueError(f"Unknown similarity measure: {similarity_measure}")

    # Get the ground truth relevant documents
    ground_truth = get_ground_truth(query)

    # Create binary arrays for evaluation
    true_labels = [1 if i in ground_truth else 0 for i in range(len(documents))]
    predicted_labels = [1 if i in top_indices else 0 for i in range(len(documents))]

    # Calculate Precision, Recall, F1-Score
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return precision, recall, f1

# Streamlit application
def main():
    st.title("Document Query Retrieval with Similarity Search")

    # Load documents
    st.sidebar.header("Document Settings")
    directory_path = st.sidebar.text_input("Directory Path", "C:/Users/sivan/smartnews/summarized_data/")

    if directory_path:
        documents = load_documents(directory_path)
        if documents:
            st.sidebar.success(f"Loaded {len(documents)} documents.")
            embeddings = create_embeddings(documents)
            np.save("embeddings.npy", embeddings)
            loaded_embeddings = np.load("embeddings.npy")
            st.sidebar.success(f"Embeddings created and loaded successfully.")

            # Query input
            query = st.text_input("Enter your query:")

            if query:
                similarity_measure = st.selectbox(
                    "Select similarity measure:",
                    ("cosine", "euclidean", "manhattan")
                )

                # Fetch top documents for the query
                top_k = st.slider("Number of Top Documents to Retrieve:", 1, 10, 5)
                if st.button("Retrieve Documents"):
                    top_documents = answer_query(documents, loaded_embeddings, query, top_k=top_k, similarity_measure=similarity_measure)

                    st.subheader("Top Retrieved Documents:")
                    for i, (filename, content, score) in enumerate(top_documents):
                        st.markdown(f"**Document {i + 1}:** {filename}")
                        st.write(f"Similarity Score: {score:.4f}")
                        st.write(content[:500] + "...")  # Show first 500 characters

                if st.button("Evaluate Model"):
                    try:
                        precision, recall, f1 = evaluate_model(documents, loaded_embeddings, query, top_k=top_k, similarity_measure=similarity_measure)
                        st.write(f"**Precision:** {precision:.4f}")
                        st.write(f"**Recall:** {recall:.4f}")
                        st.write(f"**F1-Score:** {f1:.4f}")
                    except ValueError as e:
                        st.error(e)

if __name__ == "__main__":
    main()
