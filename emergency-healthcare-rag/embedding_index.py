import pathlib
import json
import re
from typing import List, Dict, Any

def load_topics_index_mapping(path: str) -> dict: 
    with open(path) as f:
        mapping = json.load(f)
    return mapping

def load_articles(topics_dir: str, topics_json_path: str) -> dict[int, str]:
    mapping = load_topics_index_mapping(topics_json_path)
    articles_by_id: dict[int, str] = {}

    for topic_path in topics_dir.iterdir():
        if topic_path.is_dir():
            topic_name = topic_path.name     
            topic_id = mapping.get(topic_name)
            
            content_parts = []
            md_files = sorted(list(topic_path.glob('*.md')))
            
            for md_file_path in md_files:
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    content_parts.append(f.read())
            
            full_content = "\n\n---\n\n".join(content_parts)
            articles_by_id[topic_id] = full_content
    
    return articles_by_id

def preprocess_article_text(text: str) -> str:
    # Remove YAML frontmatter
    text = re.sub(r'---\n(.*?)\n---', '', text, flags=re.DOTALL) 

    # Remove the entire References section
    text = re.sub(r'## References.*', '', text, flags=re.DOTALL)
    
    # Remove other specific boilerplate sections and lines
    patterns_to_remove = [
        r'Author Information and Affiliations',
        r'#### Authors\n.*?\nLast Update:.*?\n',
        r'## Continuing Education Activity',
        r'^\*\*Objectives:\*\*.*?\n(?=##)', # Objectives block until the next H2
        r'\[Access free multiple choice questions on this topic\.\]\(.*?\)',
        r'\[Comment on this article\.\]\(.*?\)',
        r'## Review Questions',
        r'^\*\*Disclosure:\*\*.*',
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.MULTILINE)

    # Clean up markdown links
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Remove image markdown (e.g., ![...](...))
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # Remove figure captions
    text = re.sub(r'#### \[Figure\].*', '', text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def chunk_text_by_sections(topic_id: int, article_text: str) -> List[Dict[str, Any]]:
    # Split the text by '## ' which indicates a new section
    # The regex uses a positive lookahead (?=...) to keep the delimiter ('## ')
    parts = re.split(r'(?=## )', article_text)
    
    chunks = []
    
    # Handle the first part (content before the first '##', if any)
    # This is often an abstract or introductory paragraph.
    first_part = parts[0].strip()
    if first_part:
        chunks.append({
            'topic_id': topic_id,
            'section_title': 'Summary', # A generic title for the preamble
            'content': first_part
        })

    # Process the remaining parts which are the main sections
    for part in parts[1:]:
        # Find the first newline to separate title from content
        try:
            title_end_index = part.find('\n')
            # Title is the first line, cleaned of '## ' and whitespace
            title = part[3:title_end_index].strip()
            content = part[title_end_index:].strip()

            if title and content:
                chunks.append({
                    'topic_id': topic_id,
                    'section_title': title,
                    'content': content
                })
        except IndexError:
            # This part might not have a newline, skip it
            continue
            
    return chunks