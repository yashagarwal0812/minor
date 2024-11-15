from bs4 import BeautifulSoup
import regex
import string
import pandas as pd
from nltk import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import requests

# nltk.download('stopwords')
# nltk.download('punkt')


def scrape_article_content(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find the article body
        article_body = soup.find('article')
        if not article_body:
            # Fallback to <div> with specific class names
            article_body = soup.find('div', class_='main-content')

        # If both `article` and `div` tags are missing, concatenate all <p> tags
        if not article_body:
            article_content = " ".join(p.get_text() for p in soup.find_all('p'))
        else:
            # Extract text content from the article body
            paragraphs = article_body.find_all('p')
            article_content = ' '.join([para.get_text() for para in paragraphs])

        # Return cleaned-up article content if found
        return article_content.strip() if article_content else None

    except Exception as e:
        print(f"Error while scraping: {e}")
        return None





stop_words=set(stopwords.words('english'))
punc=string.punctuation

abbreviation_dict = {
    'LOL': 'laugh out loud',
    'BRB': 'be right back',
    'OMG': 'oh my god',
    'AFAIK': 'as far as I know',
    'AFK': 'away from keyboard',
    'ASAP': 'as soon as possible',
    'ATK': 'at the keyboard',
    'ATM': 'at the moment',
    'A3': 'anytime, anywhere, anyplace',
    'BAK': 'back at keyboard',
    'BBL': 'be back later',
    'BBS': 'be back soon',
    'BFN': 'bye for now',
    'B4N': 'bye for now',
    'BRB': 'be right back',
    'BRT': 'be right there',
    'BTW': 'by the way',
    'B4': 'before',
    'B4N': 'bye for now',
    'CU': 'see you',
    'CUL8R': 'see you later',
    'CYA': 'see you',
    'FAQ': 'frequently asked questions',
    'FC': 'fingers crossed',
    'FWIW': 'for what it\'s worth',
    'FYI': 'For Your Information',
    'GAL': 'get a life',
    'GG': 'good game',
    'GN': 'good night',
    'GMTA': 'great minds think alike',
    'GR8': 'great!',
    'G9': 'genius',
    'IC': 'i see',
    'ICQ': 'i seek you',
    'ILU': 'i love you',
    'IMHO': 'in my honest/humble opinion',
    'IMO': 'in my opinion',
    'IOW': 'in other words',
    'IRL': 'in real life',
    'KISS': 'keep it simple, stupid',
    'LDR': 'long distance relationship',
    'LMAO': 'laugh my a.. off',
    'LOL': 'laughing out loud',
    'LTNS': 'long time no see',
    'L8R': 'later',
    'MTE': 'my thoughts exactly',
    'M8': 'mate',
    'NRN': 'no reply necessary',
    'OIC': 'oh i see',
    'PITA': 'pain in the a..',
    'PRT': 'party',
    'PRW': 'parents are watching',
    'QPSA?': 'que pasa?',
    'ROFL': 'rolling on the floor laughing',
    'ROFLOL': 'rolling on the floor laughing out loud',
    'ROTFLMAO': 'rolling on the floor laughing my a.. off',
    'SK8': 'skate',
    'STATS': 'your sex and age',
    'ASL': 'age, sex, location',
    'THX': 'thank you',
    'TTFN': 'ta-ta for now!',
    'TTYL': 'talk to you later',
    'U': 'you',
    'U2': 'you too',
    'U4E': 'yours for ever',
    'WB': 'welcome back',
    'WTF': 'what the f...',
    'WTG': 'way to go!',
    'WUF': 'where are you from?',
    'W8': 'wait...',
    '7K': 'sick laughter',
    'TFW': 'that feeling when',
    'MFW': 'my face when',
    'MRW': 'my reaction when',
    'IFYP': 'i feel your pain',
    'LOL': 'laughing out loud',
    'TNTL': 'trying not to laugh',
    'JK': 'just kidding',
    'IDC': 'i don’t care',
    'ILY': 'i love you',
    'IMU': 'i miss you',
    'ADIH': 'another day in hell',
    'IDC': 'i don’t care',
    'ZZZ': 'sleeping, bored, tired',
    'WYWH': 'wish you were here',
    'TIME': 'tears in my eyes',
    'BAE': 'before anyone else',
    'FIMH': 'forever in my heart',
    'BSAAW': 'big smile and a wink',
    'BWL': 'bursting with laughter',
    'LMAO': 'laughing my a** off',
    'BFF': 'best friends forever',
    'CSL': 'can’t stop laughing',
}

def tokenize_text(text):
    words_list = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    words = ' '.join(' '.join(words) for words in words_list)
    return words

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def has_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return bool(soup.find())

def remove_emojis(text):
    emoji_pattern = regex.compile(r'\p{Emoji}', flags=regex.UNICODE)
    return emoji_pattern.sub('', text)

def remove_url(text):
    pattern=re.compile(r'https?://\S+|www\.S+')
    return pattern.sub(r'',text)

def remove_punc(text):
    return text.translate(str.maketrans('', '',punc))

def replace_abbreviations(text):
    for abbreviation, full_form in abbreviation_dict.items():
        text = text.replace(abbreviation, full_form)
    return text

y = pd.read_csv('trans_data.csv')

model = joblib.load('modelCate.pkl')

le = LabelEncoder()
le.fit(y['category_grouped'])

desc = """With less than a week to go for campaigning to end in Maharashtra, we are well and truly into the last stretch of an Assembly election that, to a large extent, will determine if the Opposition has what it takes to sustain the sense of hope that animated it after the Lok Sabha polls or if the BJP-led NDA will be able to establish that the parliamentary elections were an aberration.
For the BJP, both its top two leaders will be in the state. Prime Minister Narendra Modi is scheduled to address public meetings in Chimur (1 pm) in eastern Vidarbha and Solapur (4.15 pm) and Pune (6.30 pm) in western Maharashtra.
Like in some other parts of the state, the Mahayuti government is also facing a degree of unrest among farmers in Vidarbha, which is the state’s cotton belt. How the PM tackles this in his speech will be among the things to watch out for as it will send a signal to farmers in the rest of the state dealing with agrarian distress.
In western Maharashtra that has 70 Assembly constituencies, the BJP and its Mahayuti allies will look to minimise any damage in what is essentially considered an NCP-Congress stronghold. Here, the BJP is looking to strategically deploy the PM to paper over any cracks in its organisation at the ground level.
The PM’s second-in-command and Union Home Minister Amit Shah will start his day in Jharkhand’s Dhanbad district by addressing rallies in Jharia Assembly constituency (11.30 am) and Baghmara Assembly seat (1.15 pm) before flying to Maharashtra to address public meetings in Ghatkopar East in suburban Mumbai (5.30 pm) and Borivali (7.30 pm)."""
# desc = scrape_article_content('https://indianexpress.com/article/lifestyle/food-wine/are-figs-non-vegetarian-the-surprising-role-of-fig-wasps-in-pollination-9662477/')
# print(desc)
desc = desc.lower()
desc = remove_emojis(desc)
desc = remove_url(desc)
desc = remove_punc(desc)
desc = remove_stopwords(desc)
desc = replace_abbreviations(desc)
desc = tokenize_text(desc)
desc = [desc]

predicted_label_encoded = model.predict(desc)

predicted_category = le.inverse_transform(predicted_label_encoded)

print("Predicted Category:", predicted_category[0])