import requests
from bs4 import BeautifulSoup
import os
import re

# Set data directory
DATA_DIR = os.path.join('data', 'hindi')

def clean_text(text):
    """Clean the text by removing unnecessary whitespace and special characters."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters except Hindi Unicode range and basic punctuation
    text = re.sub(r'[^\u0900-\u097F\s\.,\?!]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def download_hindi_text():
    """Download Hindi text from Hindi Wikipedia featured articles."""
    # URLs of some Hindi Wikipedia featured articles
    urls = [
        'https://hi.wikipedia.org/wiki/भारत',
        'https://hi.wikipedia.org/wiki/हिन्दी',
        'https://hi.wikipedia.org/wiki/दिल्ली',
        'https://hi.wikipedia.org/wiki/महात्मा_गांधी',
        'https://hi.wikipedia.org/wiki/योग',
        'https://hi.wikipedia.org/wiki/भारतीय_संविधान',
        'https://hi.wikipedia.org/wiki/भारतीय_राष्ट्रपति',
        'https://hi.wikipedia.org/wiki/भारतीय_संविधान_सभा',
        'https://hi.wikipedia.org/wiki/भारतीय_राष्ट्रपति_चुनाव',
        'https://hi.wikipedia.org/wiki/भारतीय_राष्ट्रपति_चुनाव_2017',
        'https://hi.wikipedia.org/wiki/भारतीय_राष्ट्रपति_चुनाव_2012',
        'https://hi.wikipedia.org/wiki/भारतीय_राष्ट्रपति_चुनाव_2007',
        'https://hi.wikipedia.org/wiki/भारतीय_राष्ट्रपति_चुनाव_2002',
        'https://hi.wikipedia.org/wiki/भारतीय_राष्ट्रपति_चुनाव_1997',
    ]
    
    all_text = []
    
    for url in urls:
        try:
            print(f"Downloading from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get main content
            content = soup.find(id='mw-content-text')
            if content:
                paragraphs = content.find_all('p')
                text = ' '.join(p.get_text() for p in paragraphs)
                cleaned_text = clean_text(text)
                if cleaned_text:
                    all_text.append(cleaned_text)
        
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    # Combine all text
    combined_text = ' '.join(all_text)
    
    # Create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save to file
    with open(os.path.join(DATA_DIR, 'text.txt'), 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    print(f"\nDownloaded and saved {len(combined_text)} characters of Hindi text")
    return combined_text

if __name__ == "__main__":
    download_hindi_text() 