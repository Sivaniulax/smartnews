import os
import shutil
import langid
from newspaper import Article
import newspaper
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Ensure NLTK resources are downloaded
nltk.download('punkt_tab')

# Directory to save news articles
DATA_DIR = 'data/'
SUMMARIZED_DIR = 'summarized_data/'
MAX_FILES = 1000  # Maximum number of news articles to keep
MIN_FILE_SIZE = 2048  # Minimum file size in bytes (2 KB)

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SUMMARIZED_DIR, exist_ok=True)

# List of irrelevant phrases to filter out
IRRELEVANT_CONTENTS = [
    "Video player was slow to load content",
    "Video content never loaded",
    "Ad froze or did not finish loading",
    "Video content did not start after ad",
    "Audio on ad was too loud",
    "Other issues"
]

def extractive_summary(content, num_sentences=3):
    """Generate an extractive summary using TF-IDF."""
    sentences = nltk.sent_tokenize(content)

    # If there are fewer sentences than desired, return all
    if len(sentences) <= num_sentences:
        return content

    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between sentences
    cosine_sim = linear_kernel(vectors, vectors)

    # Rank sentences based on the sum of their cosine similarity scores
    ranked_sentences = [sentences[i] for i in cosine_sim.sum(axis=1).argsort()[::-1]]

    # Return the top 'num_sentences' sentences as the summary
    return ' '.join(ranked_sentences[:num_sentences])

def is_valid_article(article):
    """Check if the article is valid (non-empty, in English, not containing irrelevant content, and above minimum file size)."""
    if not article.text.strip():  # Check if the content is empty
        return False

    lang, _ = langid.classify(article.text)
    if lang != 'en':  # Return False if the content is not in English
        return False

    # Check for irrelevant content
    for irrelevant in IRRELEVANT_CONTENTS:
        if irrelevant.lower() in article.text.lower():  # Case insensitive check
            return False

    # Check if the article text is large enough
    if len(article.text.encode('utf-8')) <= MIN_FILE_SIZE:  # Check if the content size is less than or equal to 2 KB
        return False
            
    return True  # Return True if the article is valid

def fetch_news():
    cnn_paper = newspaper.build('https://www.cnn.com', memoize_articles=False)

    for article in cnn_paper.articles[:100]:  # Limit to 100 articles for now
        try:
            article.download()
            article.parse()

            if not is_valid_article(article):
                print(f"Invalid article skipped: {article.title}")
                continue  # Skip invalid articles

            # Create a filename from the article title
            filename = f"{DATA_DIR}{article.title.replace(' ', '_').replace('/', '_')}.txt"

            # Save the article text to a file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article.title}\n")
                f.write(f"URL: {article.url}\n")
                f.write(f"Published Date: {article.publish_date}\n")
                f.write(f"Content:\n{article.text}\n")

            print(f'Saved article: {article.title}')

            # Check if the article has already been summarized
            summary_filename = f"{SUMMARIZED_DIR}{article.title.replace(' ', '_').replace('/', '_')}_summary.txt"
            if os.path.exists(summary_filename):
                print(f'Summary already exists for article: {article.title}')
                continue  # Skip summarization if already done

            # Summarize the article using extractive summarization
            summary = extractive_summary(article.text)
            if summary.strip():  # Check if summary is not empty
                with open(summary_filename, 'w', encoding='utf-8') as sf:
                    sf.write(f"Title: {article.title}\n")
                    sf.write(f"Summary:\n{summary}\n")
                print(f'Saved summary for article: {article.title}')
            else:
                print(f'Empty summary for article: {article.title}. Summary file will not be saved.')

        except Exception as e:
            print(f"Error processing article: {e}")

    # Manage file storage
    manage_files()

def manage_files():
    """Delete older files if the number exceeds MAX_FILES and remove empty summary files."""
    # Delete old articles
    files = os.listdir(DATA_DIR)

    if len(files) > MAX_FILES:
        files.sort(key=lambda x: os.path.getctime(os.path.join(DATA_DIR, x)))
        for file in files[:-MAX_FILES]:
            os.remove(os.path.join(DATA_DIR, file))
            print(f'Deleted old article file: {file}')

    # Repeat for summarized articles
    summarized_files = os.listdir(SUMMARIZED_DIR)
    if len(summarized_files) > MAX_FILES:
        summarized_files.sort(key=lambda x: os.path.getctime(os.path.join(SUMMARIZED_DIR, x)))
        for file in summarized_files[:-MAX_FILES]:
            os.remove(os.path.join(SUMMARIZED_DIR, file))
            print(f'Deleted old summary file: {file}')
    
    # Remove empty files from the summarized directory
    for file in summarized_files:
        file_path = os.path.join(SUMMARIZED_DIR, file)
        if os.path.getsize(file_path) == 0:  # Check if file size is 0 KB
            os.remove(file_path)
            print(f'Deleted empty summary file: {file}')

if __name__ == "__main__":
    fetch_news()
