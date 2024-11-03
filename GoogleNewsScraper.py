from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import pandas as pd

def Company_News_Scraper(Company_Name):
    # Define the company name and construct the search URL
    search_url = f"https://news.google.com/search?q={Company_Name.replace(' ', '%20')}"

    # Send a request to Google News
    response = requests.get(search_url)
    if response.status_code == 200:
        print("Page fetched successfully")
    else:
        print("Failed to retrieve page")
        exit()  # Exit if page fetch fails

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Find and extract article titles
    titles = soup.find_all("a", class_="JtKRv")  # Adjust class name if needed
    title_texts = [title_tag.get_text() for title_tag in titles]

    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define a more focused reference sentence for financial relevance
    reference_sentence = (
        "Corporate financial results, quarterly earnings, company revenue, investments, "
        "economic impact, market share, stock performance, profits, losses, acquisitions, mergers, "
        "capital expenditure, cash flow, dividends, cost-cutting, debt, liabilities, assets, valuation, "
        "funding, IPO, equity, restructuring, layoffs, forecasts, analyst ratings, operating income, "
        "net income, gross profit margin, sales growth, financial outlook, earnings per share, shareholder value."
    )

    # Encode the reference sentence
    reference_embedding = model.encode(reference_sentence, convert_to_tensor=True)

    # Encode the titles
    title_embeddings = model.encode(title_texts, convert_to_tensor=True)

    # Calculate similarity scores with the reference sentence
    cosine_scores = util.cos_sim(reference_embedding, title_embeddings)[0]

    # Set a lower threshold
    threshold = 0.12  # Lowered to capture more potentially relevant titles

    # Filter titles based on the new threshold
    relevant_titles = [title_texts[i] for i in range(len(title_texts)) if cosine_scores[i] > threshold]

    # Display the relevant titles
    print("\nRelevant Titles Based on Semantic Similarity:")
    for title in relevant_titles:
        print(title)

    # Save relevant titles to a DataFrame for further use
    df = pd.DataFrame(relevant_titles, columns=["Relevant Titles"])
    print("\nFiltered Titles DataFrame:")
    print(df)
    
    return df

Company_News_Scraper("Tesla")