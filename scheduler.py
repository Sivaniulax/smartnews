from apscheduler.schedulers.blocking import BlockingScheduler
from news_collection import collect_news
#from news_summarization import summarize_news
import datetime

def scheduled_job():
    print(f"Collecting news at {datetime.datetime.now()}")
    collect_news()
    #print("Summarizing news articles...")
    #summarize_news()

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    
    # Schedule job every hour
    scheduler.add_job(scheduled_job, 'interval', hours=0.1)  
    
    print("Scheduler started. Collecting and summarizing news...")
    scheduler.start()
