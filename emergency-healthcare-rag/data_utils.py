import json
import re
import pathlib
import pickle
import logging
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


def create_clean_chunks(topic_id: int, raw_text: str) -> List[Dict[str, Any]]:
    unwanted_sections = {
        'Authors',
        'Affiliations',
        'Continuing Education Activity',
        'Review Questions',
        'References',
        'Disclosure',
        'Comment on this article.',
        'Access free multiple choice questions on this topic.' 
    }
    
    text = re.sub(r'---\n(.*?)\n---', '', raw_text, flags=re.DOTALL)
    text = re.sub(r'## References.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = text.replace('\\\'', '\'')
    text = re.sub(r'_(.*?)_', r'\1', text)

    preamble_and_content = re.split(r'(?=## )', text, maxsplit=1)
    
    if len(preamble_and_content) == 2:
        main_content = preamble_and_content[1]
    else:
        main_content = preamble_and_content[0]

    title = raw_text.split('\n')[0]
    text = f"{title}\n\n{main_content}"

    text = text.split('* \n\n  * Click here for a simplified version.')[0]
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    parts = re.split(r'(?=## )', text)
    chunks = []

    first_part_content = parts[0].replace(title, '').strip()
    if first_part_content:
        chunks.append({
            'topic_id': topic_id,
            'section_title': 'Summary',
            'content': first_part_content
        })

    for part in parts[1:]:
        try:
            title_end_index = part.find('\n')
            section_title = part[3:title_end_index].strip()
            section_content = part[title_end_index:].strip()

            if section_title and section_content and section_title not in unwanted_sections:
                chunks.append({
                    'topic_id': topic_id,
                    'section_title': section_title,
                    'content': section_content
                })
        except IndexError:
            continue
            
    return chunks


def prepare_and_save_chunks(topics_dir: pathlib.Path, topics_json_path: pathlib.Path, output_file: pathlib.Path) -> List[Dict[str, Any]]:
    name_to_id_map = load_topics_index_mapping(topics_json_path)
    master_chunks_list = []
    for topic_path in pathlib.Path(topics_dir).iterdir():
        if not topic_path.is_dir():
            continue

        topic_name = topic_path.name
        topic_id = name_to_id_map.get(topic_name)
        content_parts = []
        for md_file_path in sorted(list(topic_path.glob('*.md'))):
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content_parts.append(f.read())
        
        raw_text = "\n\n---\n\n".join(content_parts)

        if raw_text:
            chunks = create_clean_chunks(topic_id, raw_text)
            master_chunks_list.extend(chunks)


    print(f"Saving the list to '{output_file}'")
    with open(output_file, "wb") as f:
        pickle.dump(master_chunks_list, f)
    
    return master_chunks_list