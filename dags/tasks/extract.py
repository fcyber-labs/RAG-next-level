"""
Document extraction from multiple sources: S3, filesystem, URLs, PostgreSQL.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import json

logger = logging.getLogger(__name__)


def extract_from_s3(bucket: str, prefix: str) -> List[Dict[str, Any]]:
    """
    Extract documents from S3 bucket.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 key prefix to filter objects
        
    Returns:
        List of document dictionaries with content and metadata
    """
    documents = []
    
    try:
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                
                # Skip directories
                if key.endswith('/'):
                    continue
                
                # Get object content
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    content = response['Body'].read()
                    
                    # Process based on file type
                    text_content = _extract_text_from_content(
                        content=content,
                        filename=key,
                    )
                    
                    if text_content:
                        documents.append({
                            'content': text_content,
                            'source': 's3',
                            'source_uri': f's3://{bucket}/{key}',
                            'filename': Path(key).name,
                            'metadata': {
                                'bucket': bucket,
                                'key': key,
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                            }
                        })
                    
                    logger.info(f"Extracted document from S3: {key}")
                    
                except ClientError as e:
                    logger.error(f"Error reading S3 object {key}: {e}")
                    continue
        
        logger.info(f"Extracted {len(documents)} documents from S3 bucket {bucket}")
        
    except Exception as e:
        logger.error(f"Error accessing S3 bucket {bucket}: {e}")
        raise
    
    return documents


def extract_from_filesystem(path: str) -> List[Dict[str, Any]]:
    """
    Extract documents from local filesystem.
    
    Args:
        path: Directory path to scan for documents
        
    Returns:
        List of document dictionaries
    """
    documents = []
    path_obj = Path(path)
    
    if not path_obj.exists():
        logger.warning(f"Filesystem path does not exist: {path}")
        return documents
    
    # Supported file extensions
    extensions = ['.txt', '.pdf', '.md', '.html', '.json']
    
    for file_path in path_obj.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                text_content = _extract_text_from_content(
                    content=content,
                    filename=str(file_path),
                )
                
                if text_content:
                    documents.append({
                        'content': text_content,
                        'source': 'filesystem',
                        'source_uri': f'file://{file_path.absolute()}',
                        'filename': file_path.name,
                        'metadata': {
                            'path': str(file_path.absolute()),
                            'size': file_path.stat().st_size,
                            'modified': file_path.stat().st_mtime,
                        }
                    })
                
                logger.info(f"Extracted document from filesystem: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                continue
    
    logger.info(f"Extracted {len(documents)} documents from filesystem")
    return documents


def extract_from_urls(url_list_path: str) -> List[Dict[str, Any]]:
    """
    Extract documents from URLs (web scraping).
    
    Args:
        url_list_path: Path to file containing URLs (one per line)
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    if not os.path.exists(url_list_path):
        logger.warning(f"URL list file not found: {url_list_path}")
        return documents
    
    with open(url_list_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'header', 'footer']):
                script.decompose()
            
            # Extract text
            text_content = soup.get_text(separator='\n', strip=True)
            
            if text_content:
                documents.append({
                    'content': text_content,
                    'source': 'url',
                    'source_uri': url,
                    'filename': soup.title.string if soup.title else url,
                    'metadata': {
                        'url': url,
                        'title': soup.title.string if soup.title else '',
                        'content_type': response.headers.get('content-type', ''),
                    }
                })
            
            logger.info(f"Extracted document from URL: {url}")
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            continue
    
    logger.info(f"Extracted {len(documents)} documents from URLs")
    return documents


def _extract_text_from_content(content: bytes, filename: str) -> str:
    """
    Extract text from file content based on file extension.
    
    Args:
        content: File content as bytes
        filename: Name of the file (to determine type)
        
    Returns:
        Extracted text content
    """
    ext = Path(filename).suffix.lower()
    
    try:
        if ext == '.pdf':
            # Extract text from PDF
            from io import BytesIO
            pdf_file = BytesIO(content)
            reader = PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text.strip()
        
        elif ext in ['.txt', '.md']:
            # Plain text or Markdown
            return content.decode('utf-8', errors='ignore').strip()
        
        elif ext == '.html':
            # HTML content
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(['script', 'style']):
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
        
        elif ext == '.json':
            # JSON - extract all text values
            data = json.loads(content.decode('utf-8'))
            return json.dumps(data, indent=2)
        
        else:
            # Try to decode as text
            return content.decode('utf-8', errors='ignore').strip()
    
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        return ''


def extract_sources(
    sources: List[str],
    s3_bucket: str = None,
    s3_prefix: str = '',
    filesystem_path: str = None,
    url_list_path: str = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Main extraction function that orchestrates all source extractors.
    
    Args:
        sources: List of source types to extract from
        s3_bucket: S3 bucket name (if 's3' in sources)
        s3_prefix: S3 prefix filter
        filesystem_path: Local directory path
        url_list_path: Path to URL list file
        
    Returns:
        Combined list of all extracted documents
    """
    all_documents = []
    
    # Parse sources if passed as string from Airflow params
    if isinstance(sources, str):
        sources = eval(sources)  # Convert string representation to list
    
    logger.info(f"Starting document extraction from sources: {sources}")
    
    if 's3' in sources and s3_bucket:
        s3_docs = extract_from_s3(bucket=s3_bucket, prefix=s3_prefix)
        all_documents.extend(s3_docs)
    
    if 'filesystem' in sources and filesystem_path:
        fs_docs = extract_from_filesystem(path=filesystem_path)
        all_documents.extend(fs_docs)
    
    if 'urls' in sources and url_list_path:
        url_docs = extract_from_urls(url_list_path=url_list_path)
        all_documents.extend(url_docs)
    
    logger.info(f"Total documents extracted: {len(all_documents)}")
    
    # Log to metrics
    from utils.metrics_exporter import export_counter
    export_counter('documents_extracted_total', len(all_documents))
    
    return all_documents