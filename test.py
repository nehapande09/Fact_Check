import requests
import re
from bs4 import BeautifulSoup

def zenrow_search(query):
    """Search Google with ZenRow API for general trending sources (for RNN)."""
    ZENROW_API_KEY = "393c94bdd237c4cfa8e15df6c13df495987332b5"
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    zenrow_url = f"https://api.zenrows.com/v1/?apikey={ZENROW_API_KEY}&url={search_url}&js_render=true&premium_proxy=true"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(zenrow_url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.select("div.tF2Cxc a")

            # Extract & clean URLs
            clean_links = [re.sub(r"/url\?q=([^&]+).*", r"\1", result["href"]) for result in search_results if result.has_attr("href")]

            session['rnn_urls'] = clean_links[:10]  # ✅ Store for RNN
            return clean_links[:10] if clean_links else None
        else:
            print(f"ZenRow API failed with status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"ZenRow API request failed: {e}")
        return None


results = zenrow_search("Narendra Modi")
print("Results:", results)