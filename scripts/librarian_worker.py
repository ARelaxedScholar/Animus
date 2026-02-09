#!/usr/bin/env python3
"""
Animus Librarian Worker
Automates source material ingestion from Project Gutenberg and Standard Ebooks.
"""

import os
import sys
import requests
import json
import psycopg2
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def ingest_gutenberg_book(book_id, author, title):
    """Fetch and chunk a book from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"Librarian: Fetching {title} by {author}...")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Librarian: Failed to fetch book {book_id}")
        return

    text = response.text
    # Basic cleaning
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    if start_idx != -1:
        text = text[text.find("\n", start_idx) + 1:]
    
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]

    # Chunking by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]
    
    print(f"Librarian: Found {len(paragraphs)} viable chunks.")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    for chunk in paragraphs[:50]: # Limit for dev
        # TODO: Use LLM to generate thematic tags
        tags = ["philosophy", author.lower()]
        
        cur.execute(
            "INSERT INTO wisdom_library (author, title, content_chunk, thematic_tags) VALUES (%s, %s, %s, %s)",
            (author, title, chunk, tags)
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"Librarian: Ingested {title} successfully.")

if __name__ == "__main__":
    # Example ingestion: Marcus Aurelius - Meditations
    ingest_gutenberg_book("158", "Marcus Aurelius", "Meditations")
