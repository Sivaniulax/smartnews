# smartnews
#video link :  https://drive.google.com/file/d/1nWmLjRLRCDJMRJAkfGdrWP12d8dIaLZt/view?usp=sharing 
Smart News Project Documentation
(Smart News: Automated News Collection, Summarization, and Retrieval)

#Abstract
The Smart News project is an automated system designed to collect news articles, summarize them, and provide a searchable interface for users. Using Python and various NLP libraries, the project collects news from a specified source (CNN), summarizes articles, and allows users to query and retrieve relevant articles. The project aims to streamline access to relevant news by summarizing and ranking content based on similarity to user queries.


##Introduction
In today’s digital era, vast amounts of news content are generated and distributed across multiple platforms every minute. While access to information has expanded dramatically, the sheer volume of news can be overwhelming, making it challenging for users to stay updated on specific topics of interest. Moreover, traditional news consumption methods often lack efficiency, requiring users to sift through multiple articles to identify relevant information. This results in both information overload and a significant time investment.

The Smart News project addresses the challenge of information overload by providing an automated system to collect, summarize, and retrieve news articles from CNN in real time. Users can access concise summaries of news and perform search queries to quickly retrieve relevant content. The system is designed to streamline news consumption by regularly gathering articles, generating brief summaries, and ranking search results based on relevance, enabling users to stay informed without sifting through excessive information.


##Problem Statement
In the current digital age, news is generated at a high frequency from multiple sources. Users struggle to sift through vast amounts of content to find news articles relevant to their interests. The Smart News project addresses this problem by automating news collection, providing concise summaries, and implementing a query-based retrieval system to help users find and evaluate relevant information efficiently.

##Data Collection
Source: News articles are collected from CNN using the newspaper3k library. This library allows web scraping and parsing of articles, saving them as text files in a structured format.
Storage: The raw articles are saved in the data/ directory, and the summarized versions are saved in the summarized_data/ directory.
Directory Management: The project uses a file management system to limit the number of stored files and ensure storage constraints are adhered to. When the directory exceeds the specified limit of 1000 files, older articles are deleted.


##Dataset
The dataset used in this project consists of news articles collected directly from CNN’s website. Each article includes:
Title: The headline of the article.
Content: Full textual content of the article.
URL: Link to the original article.
Published Date: Date the article was published.


##Methods Used
1. Article Collection and Storage
Tool: newspaper3k library.
Process: Articles are collected from CNN, parsed, and saved in the data/ directory. The collection is automated using APScheduler to ensure continuous updates at specified intervals.
File Management: manage_files() function limits the total stored articles to 1000 by deleting the oldest files.
2. Extractive Summarization
Tool: TF-IDF (Term Frequency-Inverse Document Frequency) with scikit-learn.
Process: Sentences within each article are ranked based on their TF-IDF scores and cosine similarity. The top-ranked sentences are selected to form a summary.
Output: Summarized articles are saved in the summarized_data/ directory.
3. Query-Based Document Retrieval
Tool: Sentence Transformers (sentence-transformers library) and scikit-learn for similarity measures.
Process: A pre-trained embedding model (all-MiniLM-L6-v2) generates embeddings for each document and query. Similarity scores are calculated using three different similarity measures: cosine similarity, Euclidean distance, and Manhattan distance.
Output: The system retrieves and ranks the top-K most relevant articles based on the user’s query.
4. Model Evaluation
Evaluation Metrics: Precision, Recall, and F1-score.
Ground Truth: A manually curated list of relevant articles for each query is used to calculate evaluation metrics.
Process: Precision, Recall, and F1-score are calculated based on the relevance of retrieved articles compared to the ground truth. This helps in assessing the accuracy of the document retrieval system.



##Implementation Details
Project Structure
scheduler.py - Automates the news collection process using APScheduler to schedule the collection of news every 6 minutes.
Main Function: scheduled_job(), which calls the collect_news() function to gather articles.
Interval: Configured to run every 0.1 hours (6 minutes).
news_collection.py
Function: collect_news() collects articles, filters out irrelevant content, and saves articles as text files in data/.
Function: manage_files() ensures storage constraints are managed by deleting older files when the limit exceeds 1000 files.
news_summarization.py
Function: extractive_summary() uses TF-IDF to rank and select key sentences from each article.
Function: fetch_news() collects, validates, and summarizes articles. Valid summaries are saved in summarized_data/.
Function: manage_files() removes empty summary files and manages file limits in summarized_data/.
main.py
Document Loading: load_documents() loads all summarized articles from summarized_data/.
Embeddings: create_embeddings() generates embeddings for each document using Sentence Transformers.
Document Retrieval: answer_query() ranks and retrieves the top-K documents based on similarity to the query.
Evaluation: evaluate_model() calculates Precision, Recall, and F1-score to assess retrieval accuracy.


##Model Evaluation and Results
The model’s performance is evaluated based on how accurately it retrieves relevant articles for a given query using Precision, Recall, and F1-score.
Precision: The fraction of relevant documents among the retrieved documents.
Recall: The fraction of relevant documents that were successfully retrieved.
F1-Score: The harmonic mean of Precision and Recall, offering a balanced measure of both metrics.
Results:
Evaluation results can vary based on the similarity measure (cosine, Euclidean, or Manhattan). Generally:
Cosine Similarity: Often produces the best results, as it captures the angle between vectors, making it effective for textual similarity.
Euclidean and Manhattan: May offer alternative perspectives but often produce lower F1-scores compared to cosine similarity in this text-based retrieval task.


##Conclusion
The Smart News project automates the collection, summarization, and retrieval of news articles, offering users a streamlined and interactive experience for staying informed on relevant topics. Through continuous improvement and expansion, this project has the potential to become a comprehensive tool for real-time news analysis and discovery.

