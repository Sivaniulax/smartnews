import os
import newspaper
from newspaper import Article
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')  

# Directory to save news articles
DATA_DIR = 'data/'
MAX_FILES = 1000  # Maximum number of news articles to keep

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)

def collect_news():
    """Collect news articles from the source."""
    cnn_paper = newspaper.build('https://www.cnn.com', memoize_articles=False)

    for article in cnn_paper.articles[:100]:  # Limit to 100 articles for now
        try:
            article.download()
            article.parse()
            
            # Create a filename from the article title
            filename = f"{DATA_DIR}{article.title.replace(' ', '_').replace('/', '_')}.txt"
            
            # Save the article text to a file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article.title}\n")
                f.write(f"URL: {article.url}\n")
                f.write(f"Published Date: {article.publish_date}\n")
                f.write(f"Content:\n{article.text}\n")
                
            print(f'Saved article: {article.title}')
            
        except Exception as e:
            print(f"Error processing article: {e}")

    # Manage file storage
    manage_files()

def manage_files():
    """Delete older files if the number exceeds MAX_FILES."""
    files = os.listdir(DATA_DIR)
    
    if len(files) > MAX_FILES:
        # Sort files by creation time
        files.sort(key=lambda x: os.path.getctime(os.path.join(DATA_DIR, x)))
        
        # Delete the oldest files
        for file in files[:-MAX_FILES]:
            os.remove(os.path.join(DATA_DIR, file))
            print(f'Deleted old article file: {file}')

if __name__ == "__main__":
    collect_news()
