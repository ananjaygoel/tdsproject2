# agent.py
"""
Universal LLM-powered quiz solver agent.
Handles: PDFs, CSVs, JSON, images, web scraping, code execution, and more.
Uses aipipe.org for LLM access (OpenAI-compatible API).
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import Optional, Any, List, Dict
from urllib.parse import urljoin, urlparse

import httpx
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from dotenv import load_dotenv

# PDF support
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# Image support
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

load_dotenv()

# ------- Configuration -------
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
# Use a capable model - gpt-4.1-mini is better for complex reasoning
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
# Vision model for images
VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4.1-mini")
TIME_LIMIT_SECONDS = 170  # Leave 10s buffer from 3 min limit
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")


def run_agent(start_url: str, email: Optional[str] = None, secret: Optional[str] = None):
    """
    Entrypoint expected by FastAPI BackgroundTasks.
    Schedules the async worker safely whether called from an event loop or not.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        def _target():
            try:
                asyncio.run(_run_agent(start_url, email, secret))
            except Exception:
                logger.exception("[agent] run in thread crashed")
        t = threading.Thread(target=_target, daemon=True)
        t.start()
    else:
        asyncio.create_task(_run_agent(start_url, email, secret))


async def _run_agent(start_url: str, email: Optional[str], secret: Optional[str]):
    """Main async worker - visits quiz URLs, solves tasks, submits answers."""
    logger.info(f"[agent] starting worker for url={start_url} email={email}")

    start_time = time.monotonic()
    deadline = start_time + TIME_LIMIT_SECONDS
    current_url = start_url

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context()
        page = await context.new_page()

        try:
            while current_url:
                if time.monotonic() > deadline:
                    logger.warning("[agent] reached time limit, stopping")
                    break

                logger.info(f"[agent] visiting {current_url}")
                
                # Load the page with JS rendering
                try:
                    await page.goto(current_url, timeout=60000, wait_until="networkidle")
                except PlaywrightTimeoutError:
                    try:
                        await page.goto(current_url, timeout=30000, wait_until="domcontentloaded")
                    except Exception as e:
                        logger.exception(f"[agent] page.goto failed: {e}")
                        break

                await asyncio.sleep(1)  # Let JS execute

                # Get page content
                page_text = await _safe_get_text(page)
                page_html = await _safe_get_html(page)
                
                logger.info(f"[agent] page text preview: {page_text[:500]}...")

                # Decode any base64/atob content
                decoded_content = _decode_atob_content(page_html)
                if decoded_content:
                    page_text = page_text + "\n\n[DECODED CONTENT]:\n" + decoded_content
                    logger.info(f"[agent] decoded content: {decoded_content[:300]}...")

                # Extract submit URL dynamically
                submit_url = _extract_submit_url(page_text, page_html, current_url)
                logger.info(f"[agent] submit URL: {submit_url}")

                # Extract links
                links = await _extract_links(page)
                
                # Collect all available data from the page
                collected_data = await _collect_page_data(
                    current_url=current_url,
                    page=page,
                    page_text=page_text,
                    page_html=page_html,
                    links=links
                )
                
                # Solve the task
                answer = await _solve_task(
                    current_url=current_url,
                    page_text=page_text,
                    page_html=page_html,
                    links=links,
                    collected_data=collected_data,
                    page=page,
                    email=email
                )
                
                if answer is None:
                    logger.warning("[agent] could not determine answer, stopping")
                    break

                logger.info(f"[agent] submitting answer: {str(answer)[:200]}")

                # Submit answer
                resp = await _submit_answer(submit_url, email, secret, current_url, answer)
                
                if not resp:
                    logger.error("[agent] submit returned no response")
                    break

                logger.info(f"[agent] submit response: {resp}")

                if resp.get("correct") is True:
                    logger.info("[agent] answer correct!")
                    next_url = resp.get("url")
                    if next_url:
                        current_url = next_url
                    else:
                        logger.info("[agent] quiz complete!")
                        break
                else:
                    reason = resp.get("reason", "unknown")
                    logger.warning(f"[agent] answer incorrect: {reason}")
                    next_url = resp.get("url")
                    if next_url:
                        current_url = next_url
                    else:
                        break

        except Exception:
            logger.exception("[agent] unexpected error")
        finally:
            await context.close()
            await browser.close()

    logger.info("[agent] finished worker")


# ============== DATA COLLECTION ==============

async def _collect_page_data(
    current_url: str,
    page,
    page_text: str,
    page_html: str,
    links: List[str]
) -> Dict[str, Any]:
    """
    Collect all available data from the page:
    - CSV files
    - JSON files
    - PDF files
    - Images
    - Scraped sub-pages
    - API endpoints
    """
    data = {
        "csv_data": [],
        "json_data": [],
        "pdf_data": [],
        "image_data": [],
        "api_data": [],
        "scraped_pages": [],
    }
    
    # Categorize links by type
    for link in links:
        if not link:
            continue
        link_lower = link.lower()
        full_url = link if link.startswith('http') else urljoin(current_url, link)
        
        if '.csv' in link_lower:
            csv_content = await _download_and_parse_csv(full_url)
            if csv_content:
                data["csv_data"].append({"url": full_url, "data": csv_content})
                
        elif '.json' in link_lower:
            json_content = await _download_and_parse_json(full_url)
            if json_content:
                data["json_data"].append({"url": full_url, "data": json_content})
                
        elif '.pdf' in link_lower:
            pdf_content = await _download_and_parse_pdf(full_url)
            if pdf_content:
                data["pdf_data"].append({"url": full_url, "data": pdf_content})
                
        elif any(ext in link_lower for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
            img_data = await _download_image_as_base64(full_url)
            if img_data:
                data["image_data"].append({"url": full_url, "base64": img_data})
    
    # Also check for URLs in HTML that might not be in anchor tags
    url_patterns = [
        (r'(https?://[^\s<>"\']+\.csv)', "csv"),
        (r'(https?://[^\s<>"\']+\.json)', "json"),
        (r'(https?://[^\s<>"\']+\.pdf)', "pdf"),
    ]
    
    for pattern, file_type in url_patterns:
        for match in re.finditer(pattern, page_html, re.I):
            url = match.group(1)
            # Check if already processed
            existing_urls = [d["url"] for d in data[f"{file_type}_data"]]
            if url not in existing_urls:
                if file_type == "csv":
                    content = await _download_and_parse_csv(url)
                    if content:
                        data["csv_data"].append({"url": url, "data": content})
                elif file_type == "json":
                    content = await _download_and_parse_json(url)
                    if content:
                        data["json_data"].append({"url": url, "data": content})
                elif file_type == "pdf":
                    content = await _download_and_parse_pdf(url)
                    if content:
                        data["pdf_data"].append({"url": url, "data": content})
    
    # Check for scrape tasks
    scrape_match = re.search(r'[Ss]crape\s+(/[^\s]+)', page_text)
    if scrape_match:
        scrape_url = urljoin(current_url, scrape_match.group(1))
        try:
            await page.goto(scrape_url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(1)
            scraped_text = await _safe_get_text(page)
            data["scraped_pages"].append({"url": scrape_url, "text": scraped_text})
            logger.info(f"[agent] scraped {scrape_url}: {scraped_text[:200]}...")
        except Exception as e:
            logger.warning(f"[agent] scrape failed: {e}")
    
    # Fetch API endpoints mentioned in the page
    api_data = await _fetch_api_endpoints(page_text, page_html)
    data["api_data"] = api_data
    
    return data


async def _fetch_api_endpoints(page_text: str, page_html: str) -> List[Dict]:
    """
    Find and fetch API endpoints mentioned in the page.
    Handles pagination, auth headers, and multiple endpoints.
    """
    api_results = []
    
    # Find API endpoint URLs
    api_pattern = r'https?://[^\s<>"\']+/api/[^\s<>"\']*'
    api_urls = set(re.findall(api_pattern, page_html, re.I))
    
    # Also check for code blocks
    code_pattern = r'<code[^>]*>([^<]*https?://[^\s<>"\']+/api/[^\s<>"\']*[^<]*)</code>'
    for match in re.finditer(code_pattern, page_html, re.I):
        url = match.group(1).strip()
        api_urls.add(url)
    
    # Extract auth headers from page text
    headers = {}
    header_match = re.search(r'header\s*[`<>"\']*(X-[A-Za-z-]+)[`<>"\']*\s*(?:with\s*value|:)\s*[`<>"\']*([a-zA-Z0-9-]+)', page_text, re.I)
    if header_match:
        headers[header_match.group(1)] = header_match.group(2)
        logger.info(f"[agent] found auth header: {header_match.group(1)}={header_match.group(2)}")
    
    for url in api_urls:
        logger.info(f"[agent] fetching API: {url}")
        
        # Check if pagination is needed
        if 'page=' in url or 'pagination' in page_text.lower():
            all_data = await _fetch_paginated_api(url, headers)
            if all_data:
                api_results.append({
                    "url": url,
                    "paginated": True,
                    "data": all_data,
                    "count": len(all_data)
                })
        else:
            # Single fetch
            result = await _fetch_api(url, headers)
            if result is not None:
                api_results.append({
                    "url": url,
                    "paginated": False,
                    "data": result
                })
    
    return api_results


async def _fetch_api(url: str, headers: Dict = None) -> Optional[Any]:
    """Fetch a single API endpoint."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers or {})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"[agent] API fetch failed {url}: {e}")
        return None


async def _fetch_paginated_api(base_url: str, headers: Dict = None) -> List[Any]:
    """Fetch all pages of a paginated API."""
    all_data = []
    page = 1
    max_pages = 100  # Safety limit
    
    # Determine base URL for pagination
    if '?' in base_url:
        if 'page=' in base_url:
            # Replace existing page param
            base_for_pagination = re.sub(r'page=\d+', 'page={}', base_url)
        else:
            base_for_pagination = base_url + '&page={}'
    else:
        base_for_pagination = base_url + '?page={}'
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            while page <= max_pages:
                url = base_for_pagination.format(page)
                resp = await client.get(url, headers=headers or {})
                resp.raise_for_status()
                data = resp.json()
                
                # Check for empty response
                if not data or (isinstance(data, list) and len(data) == 0):
                    break
                
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
                
                page += 1
                
        logger.info(f"[agent] fetched {page-1} pages, {len(all_data)} items from {base_url}")
        return all_data
        
    except Exception as e:
        logger.warning(f"[agent] paginated fetch failed: {e}")
        return all_data if all_data else []


async def _download_and_parse_csv(url: str) -> Optional[Dict]:
    """Download and parse CSV file."""
    try:
        content = await _download_file(url)
        if not content:
            return None
        
        # Detect header
        buf = io.BytesIO(content)
        first_line = buf.readline().decode('utf-8', errors='ignore').strip()
        buf.seek(0)
        
        try:
            float(first_line.split(',')[0])
            df = pd.read_csv(buf, header=None)
            has_header = False
        except ValueError:
            df = pd.read_csv(io.BytesIO(content))
            has_header = True
        
        logger.info(f"[agent] CSV {url}: shape={df.shape}, header={has_header}")
        
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "has_header": has_header,
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "head": df.head(10).to_dict('records'),
            "describe": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            "raw_df": df,  # Keep for calculations
        }
    except Exception as e:
        logger.warning(f"[agent] CSV parse failed {url}: {e}")
        return None


async def _download_and_parse_json(url: str) -> Optional[Any]:
    """Download and parse JSON file."""
    try:
        content = await _download_file(url)
        if not content:
            return None
        data = json.loads(content.decode('utf-8'))
        logger.info(f"[agent] JSON {url}: type={type(data).__name__}")
        return data
    except Exception as e:
        logger.warning(f"[agent] JSON parse failed {url}: {e}")
        return None


async def _download_and_parse_pdf(url: str) -> Optional[Dict]:
    """Download and extract text/tables from PDF."""
    if not HAS_PDF:
        logger.warning("[agent] pdfplumber not available")
        return None
    
    try:
        content = await _download_file(url)
        if not content:
            return None
        
        # Write to temp file for pdfplumber
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            result = {"pages": [], "tables": [], "full_text": ""}
            
            with pdfplumber.open(temp_path) as pdf:
                for i, pg in enumerate(pdf.pages):
                    page_text = pg.extract_text() or ""
                    result["pages"].append({"page": i + 1, "text": page_text})
                    result["full_text"] += page_text + "\n"
                    
                    # Extract tables
                    tables = pg.extract_tables()
                    for table in tables:
                        if table:
                            # Convert to DataFrame for easier processing
                            df = pd.DataFrame(table[1:], columns=table[0] if table[0] else None)
                            result["tables"].append({
                                "page": i + 1,
                                "data": df.to_dict('records'),
                                "columns": list(df.columns),
                                "raw_df": df,
                            })
            
            logger.info(f"[agent] PDF {url}: {len(result['pages'])} pages, {len(result['tables'])} tables")
            return result
            
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        logger.exception(f"[agent] PDF parse failed {url}: {e}")
        return None


async def _download_image_as_base64(url: str) -> Optional[str]:
    """Download image and convert to base64."""
    try:
        content = await _download_file(url)
        if not content:
            return None
        return base64.b64encode(content).decode('utf-8')
    except Exception as e:
        logger.warning(f"[agent] image download failed {url}: {e}")
        return None


# ============== TASK SOLVING ==============

async def _solve_task(
    current_url: str,
    page_text: str,
    page_html: str,
    links: List[str],
    collected_data: Dict[str, Any],
    page,
    email: str
) -> Optional[Any]:
    """
    Universal task solver.
    Uses pattern matching for known patterns, LLM for everything else.
    """
    
    # === QUICK PATTERNS ===
    
    # 1. Demo page (accepts any answer)
    if "anything you want" in page_text.lower():
        logger.info("[agent] demo page detected")
        return "demo"
    
    # 2. Secret code in scraped data
    for scraped in collected_data.get("scraped_pages", []):
        text = scraped.get("text", "")
        secret_match = re.search(r'[Ss]ecret\s*[Cc]ode\s*(?:is)?[:\s]+([a-zA-Z0-9]+)', text)
        if secret_match:
            code = secret_match.group(1)
            logger.info(f"[agent] found secret code in scraped page: {code}")
            return code
    
    # 3. Secret code in page text
    secret_match = re.search(r'[Ss]ecret\s*[Cc]ode\s*(?:is|=|:)\s*([a-zA-Z0-9]+)', page_text)
    if secret_match:
        logger.info(f"[agent] found secret code: {secret_match.group(1)}")
        return secret_match.group(1)
    
    # 4. DOM Parsing - hidden-key with reversed text
    hidden_key_match = re.search(r'class\s*=\s*["\']hidden-key["\'][^>]*>([^<]+)<', page_html, re.I)
    if hidden_key_match:
        reversed_text = hidden_key_match.group(1).strip()
        unreversed = reversed_text[::-1]  # Reverse the string
        logger.info(f"[agent] found hidden-key reversed text: {reversed_text} -> {unreversed}")
        return unreversed
    
    # 5. JS Execution - extract and evaluate simple JS
    if 'script' in page_text.lower() or 'generated by' in page_text.lower():
        js_answer = _evaluate_js_from_html(page_html)
        if js_answer is not None:
            logger.info(f"[agent] JS execution result: {js_answer}")
            return js_answer
    
    # 6. Data cleaning with dirty data - compute sum of valid numeric prices
    if 'dirty' in page_text.lower() and 'clean' in page_text.lower() and 'sum' in page_text.lower():
        for api_item in collected_data.get("api_data", []):
            data = api_item.get("data", [])
            if isinstance(data, list):
                total = _sum_valid_prices(data)
                if total is not None:
                    logger.info(f"[agent] dirty data sum: {total}")
                    return total
    
    # 7. Data pipeline - join users/orders/products for gold tier
    if 'gold' in page_text.lower() and 'tier' in page_text.lower() and collected_data.get("api_data"):
        pipeline_result = _calculate_gold_tier_total(collected_data["api_data"])
        if pipeline_result is not None:
            logger.info(f"[agent] gold tier pipeline result: {pipeline_result}")
            return pipeline_result
    
    # 8. UV http command - construct exact command string
    if 'uv http get' in page_text.lower() or 'uv run' in page_text.lower():
        uv_cmd = _construct_uv_command(page_text, email)
        if uv_cmd:
            logger.info(f"[agent] UV command: {uv_cmd}")
            return uv_cmd
    
    # 9. Heatmap/dominant color - count pixels
    if 'heatmap' in page_text.lower() or 'dominant' in page_text.lower() or 'most frequent' in page_text.lower():
        for img in collected_data.get("image_data", []):
            color = await _get_dominant_color(img["url"])
            if color:
                logger.info(f"[agent] dominant color: {color}")
                return color
    
    # 10. CSV to JSON normalization
    if 'normalize' in page_text.lower() and 'json' in page_text.lower() and collected_data.get("csv_data"):
        json_result = _normalize_csv_to_json(collected_data["csv_data"][0]["data"]["raw_df"], page_text)
        if json_result:
            logger.info(f"[agent] normalized JSON: {json_result[:100]}...")
            return json_result
    
    # 11. GitHub API tree counting
    if 'github' in page_text.lower() and 'tree' in page_text.lower():
        gh_result = await _count_github_files(page_text, collected_data, email)
        if gh_result is not None:
            logger.info(f"[agent] GitHub tree count: {gh_result}")
            return gh_result
    
    # === DATA-BASED SOLVING ===
    
    # Build context for LLM (include both visible text and HTML for DOM tasks)
    context_parts = [f"PAGE URL: {current_url}", f"PAGE CONTENT:\n{page_text[:3000]}"]
    
    # Include HTML for DOM parsing tasks OR JS execution tasks
    if any(kw in page_text.lower() for kw in ['dom', 'hidden', 'reversed', 'script', 'javascript', 'js execution']):
        context_parts.append(f"\nPAGE HTML (includes scripts):\n{page_html[:5000]}")
    
    # Add CSV data context
    for csv_item in collected_data.get("csv_data", []):
        csv_info = csv_item["data"]
        context_parts.append(f"\nCSV FILE: {csv_item['url']}")
        context_parts.append(f"Shape: {csv_info['shape']}, Columns: {csv_info['columns']}")
        context_parts.append(f"First rows: {json.dumps(csv_info['head'][:5], default=str)}")
        if csv_info.get('describe'):
            context_parts.append(f"Stats: {json.dumps(csv_info['describe'], default=str)[:500]}")
    
    # Add JSON data context
    for json_item in collected_data.get("json_data", []):
        context_parts.append(f"\nJSON FILE: {json_item['url']}")
        context_parts.append(f"Data: {json.dumps(json_item['data'], default=str)[:2000]}")
    
    # Add PDF data context
    for pdf_item in collected_data.get("pdf_data", []):
        pdf_info = pdf_item["data"]
        context_parts.append(f"\nPDF FILE: {pdf_item['url']}")
        context_parts.append(f"Text: {pdf_info['full_text'][:2000]}")
        for table in pdf_info.get("tables", [])[:3]:
            context_parts.append(f"Table on page {table['page']}: {json.dumps(table['data'][:10], default=str)}")
    
    # Add image info
    for img_item in collected_data.get("image_data", []):
        context_parts.append(f"\nIMAGE: {img_item['url']} (available for vision analysis)")
    
    # Add API data context
    for api_item in collected_data.get("api_data", []):
        context_parts.append(f"\nAPI ENDPOINT: {api_item['url']}")
        if api_item.get("paginated"):
            context_parts.append(f"Paginated: {api_item['count']} total items")
        api_data_str = json.dumps(api_item['data'], default=str)
        # Limit size but show enough for analysis
        if len(api_data_str) > 3000:
            context_parts.append(f"Data (truncated): {api_data_str[:3000]}...")
        else:
            context_parts.append(f"Data: {api_data_str}")
    
    full_context = "\n".join(context_parts)
    
    # === TRY SPECIFIC PATTERNS ===
    
    # Pattern: Find item by ID in paginated data
    id_match = re.search(r'(?:item|record)\s+with\s+ID\s+(\d+)', page_text, re.I)
    if id_match and collected_data.get("api_data"):
        target_id = int(id_match.group(1))
        for api_item in collected_data["api_data"]:
            data = api_item.get("data", [])
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("id") == target_id:
                        # Return the name or relevant field
                        if "name" in item:
                            logger.info(f"[agent] found item ID {target_id}: {item['name']}")
                            return item["name"]
    
    # Pattern: Find highest/lowest value
    highest_match = re.search(r'highest\s+(\w+)', page_text, re.I)
    if highest_match and collected_data.get("api_data"):
        field = highest_match.group(1).lower()
        for api_item in collected_data["api_data"]:
            data = api_item.get("data", [])
            if isinstance(data, list) and data:
                # Find item with highest value for the field
                max_item = None
                max_val = float('-inf')
                for item in data:
                    if isinstance(item, dict):
                        for key, val in item.items():
                            if key.lower() == field or field in key.lower():
                                try:
                                    num_val = float(val)
                                    if num_val > max_val:
                                        max_val = num_val
                                        max_item = item
                                except (TypeError, ValueError):
                                    pass
                if max_item:
                    # Return city/name/id
                    for key in ["city", "name", "id"]:
                        if key in max_item:
                            logger.info(f"[agent] highest {field}: {max_item}")
                            return max_item[key]
    
    # === TRY SIMPLE CALCULATIONS ===
    
    # Check for cutoff/sum pattern with CSV
    cutoff_match = re.search(r'[Cc]utoff[:\s]*(\d+)', page_text)
    if cutoff_match and collected_data.get("csv_data"):
        cutoff = int(cutoff_match.group(1))
        result = _calculate_csv_sum(collected_data["csv_data"][0]["data"]["raw_df"], cutoff, page_text)
        if result is not None:
            logger.info(f"[agent] CSV calculation result: {result}")
            return result
    
    # Check for PDF table calculations
    if collected_data.get("pdf_data"):
        pdf_result = await _solve_pdf_task(collected_data["pdf_data"], page_text)
        if pdf_result is not None:
            return pdf_result
    
    # === USE LLM FOR COMPLEX TASKS ===
    
    # Check if we have images - use vision model
    if collected_data.get("image_data"):
        answer = await _solve_with_vision(
            context=full_context,
            images=collected_data["image_data"],
            task_text=page_text
        )
        if answer:
            return answer
    
    # Use LLM with code execution for complex calculations
    answer = await _solve_with_llm_and_code(
        context=full_context,
        task_text=page_text,
        collected_data=collected_data
    )
    
    return answer


def _calculate_csv_sum(df: pd.DataFrame, cutoff: int, task_text: str) -> Optional[int]:
    """Calculate sum/count/etc from CSV with cutoff."""
    try:
        task_lower = task_text.lower()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            # Try first column
            target_col = df.columns[0]
            values = pd.to_numeric(df[target_col], errors='coerce').dropna()
        else:
            target_col = numeric_cols[0]
            values = df[target_col].dropna()
        
        logger.info(f"[agent] CSV: using column {target_col}, {len(values)} values")
        
        # Apply cutoff
        if cutoff > 0:
            filtered = values[values >= cutoff]
            logger.info(f"[agent] CSV: {len(filtered)} values >= {cutoff}")
        else:
            filtered = values
        
        # Determine operation
        if 'count' in task_lower or 'how many' in task_lower:
            result = len(filtered)
        elif 'average' in task_lower or 'mean' in task_lower:
            result = filtered.mean()
        elif 'max' in task_lower:
            result = filtered.max()
        elif 'min' in task_lower:
            result = filtered.min()
        else:
            result = filtered.sum()
        
        return int(result) if pd.notna(result) else None
        
    except Exception as e:
        logger.warning(f"[agent] CSV calculation failed: {e}")
        return None


async def _solve_pdf_task(pdf_data: List[Dict], task_text: str) -> Optional[Any]:
    """Solve tasks involving PDF data."""
    try:
        # Combine all PDF text and tables
        all_text = ""
        all_tables = []
        
        for pdf_item in pdf_data:
            pdf_info = pdf_item["data"]
            all_text += pdf_info["full_text"] + "\n"
            all_tables.extend(pdf_info.get("tables", []))
        
        # Check for sum/value extraction in tables
        task_lower = task_text.lower()
        
        if 'sum' in task_lower and all_tables:
            # Try to find the relevant column and sum it
            for table in all_tables:
                df = table["raw_df"]
                for col in df.columns:
                    col_str = str(col).lower()
                    if 'value' in col_str or 'amount' in col_str or 'total' in col_str:
                        values = pd.to_numeric(df[col], errors='coerce').dropna()
                        result = int(values.sum())
                        logger.info(f"[agent] PDF table sum: {result}")
                        return result
        
        # Use LLM for complex PDF tasks
        prompt = f"""Analyze this PDF content and answer the question.

TASK:
{task_text[:1500]}

PDF TEXT:
{all_text[:3000]}

PDF TABLES:
{json.dumps([{"page": t["page"], "data": t["data"][:10]} for t in all_tables[:3]], default=str)[:2000]}

Return ONLY the answer (number, string, or JSON). No explanation."""

        return await _call_llm(prompt)
        
    except Exception as e:
        logger.exception(f"[agent] PDF task solving failed: {e}")
        return None


async def _solve_with_vision(context: str, images: List[Dict], task_text: str) -> Optional[Any]:
    """Use vision model to analyze images."""
    if not AIPIPE_TOKEN:
        return None
    
    try:
        # Build message with images
        content = [{"type": "text", "text": f"""Analyze the image(s) and answer the question.

TASK:
{task_text[:2000]}

CONTEXT:
{context[:2000]}

Return ONLY the answer. No explanation."""}]
        
        # Add images (limit to first 3)
        for img in images[:3]:
            # Detect image type from URL
            url = img["url"].lower()
            media_type = "image/png"
            if '.jpg' in url or '.jpeg' in url:
                media_type = "image/jpeg"
            elif '.gif' in url:
                media_type = "image/gif"
            elif '.webp' in url:
                media_type = "image/webp"
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{img['base64']}"}
            })
        
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                AIPIPE_URL,
                headers={
                    "Authorization": f"Bearer {AIPIPE_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": VISION_MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 500,
                    "temperature": 0
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            if "choices" in data and data["choices"]:
                answer = data["choices"][0]["message"]["content"].strip()
                logger.info(f"[agent] vision response: {answer[:200]}")
                return _parse_llm_answer(answer)
                
    except Exception as e:
        logger.exception(f"[agent] vision analysis failed: {e}")
    
    return None


async def _solve_with_llm_and_code(
    context: str,
    task_text: str,
    collected_data: Dict
) -> Optional[Any]:
    """
    Use LLM to understand the task, optionally generate and execute code.
    """
    if not AIPIPE_TOKEN:
        return None
    
    # First, ask LLM to analyze and potentially provide code
    prompt = f"""You are solving a data analysis quiz task. Analyze the context and provide the answer.

{context[:5000]}

TASK INSTRUCTIONS (from the page):
{task_text[:2000]}

IMPORTANT INSTRUCTIONS:
1. If you can directly determine the answer, provide it.
2. If calculation is needed and data is provided, compute it.
3. Return ONLY the final answer - a number, string, or JSON.
4. DO NOT explain, DO NOT wrap in quotes unless it's a string answer.
5. If the answer is a number, return just the number.
6. If asked for a sum of values >= cutoff, compute that sum.

ANSWER:"""

    try:
        answer = await _call_llm(prompt)
        if answer:
            parsed = _parse_llm_answer(answer)
            
            # If LLM says it needs to compute or gives Python code, try to execute it
            if isinstance(parsed, str) and ('compute' in parsed.lower() or 'calculate' in parsed.lower() or 'python' in parsed.lower()):
                # Ask LLM for Python code
                code_result = await _execute_llm_code(context, task_text, collected_data)
                if code_result is not None:
                    return code_result
            
            return parsed
            
    except Exception as e:
        logger.exception(f"[agent] LLM solving failed: {e}")
    
    return None


async def _execute_llm_code(context: str, task_text: str, collected_data: Dict) -> Optional[Any]:
    """Ask LLM to generate Python code and execute it."""
    
    # Prepare data info for code generation
    data_info = []
    for csv_item in collected_data.get("csv_data", []):
        url = csv_item["url"]
        data_info.append(f"CSV available at: {url}")
    for json_item in collected_data.get("json_data", []):
        url = json_item["url"]
        data_info.append(f"JSON available at: {url}")
    
    code_prompt = f"""Write Python code to solve this task. The code should print ONLY the final answer.

TASK:
{task_text[:1500]}

DATA AVAILABLE:
{chr(10).join(data_info)}

Write complete Python code that:
1. Downloads the data using requests
2. Processes it (pandas for CSV, json for JSON)
3. Computes the answer
4. Prints ONLY the final numeric/string answer

```python
import requests
import pandas as pd
import json

# Your code here
```

Return ONLY the Python code block, nothing else."""

    try:
        code_response = await _call_llm(code_prompt)
        if not code_response:
            return None
        
        # Extract code from response
        code = _extract_code(code_response)
        if not code:
            return None
        
        logger.info(f"[agent] executing generated code:\n{code[:500]}...")
        
        # Execute the code
        result = await _run_python_code(code)
        if result and result.get("return_code") == 0:
            stdout = result["stdout"].strip()
            if stdout:
                logger.info(f"[agent] code output: {stdout}")
                # Try to parse as number
                try:
                    return int(stdout)
                except ValueError:
                    try:
                        return float(stdout)
                    except ValueError:
                        return stdout
        else:
            logger.warning(f"[agent] code execution failed: {result}")
            
    except Exception as e:
        logger.exception(f"[agent] code execution error: {e}")
    
    return None


def _extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    # Look for code blocks
    code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    code_match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no code blocks, assume entire response is code
    if 'import' in response or 'print(' in response:
        return response.strip()
    
    return None


async def _run_python_code(code: str) -> Optional[Dict]:
    """Execute Python code in a subprocess."""
    try:
        # Create temp directory
        os.makedirs("LLMFiles", exist_ok=True)
        
        # Write code to file
        code_path = os.path.join("LLMFiles", "runner.py")
        with open(code_path, "w") as f:
            f.write(code)
        
        # Run with timeout
        proc = await asyncio.create_subprocess_exec(
            "python3", code_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="LLMFiles"
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return {
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "return_code": proc.returncode
            }
        except asyncio.TimeoutError:
            proc.kill()
            return {"stdout": "", "stderr": "Timeout", "return_code": -1}
            
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "return_code": -1}


# ============== SPECIFIC TASK HANDLERS ==============

def _evaluate_js_from_html(page_html: str) -> Optional[Any]:
    """Extract and evaluate simple JavaScript from HTML."""
    try:
        # Find script content
        script_match = re.search(r'<script[^>]*>(.*?)</script>', page_html, re.DOTALL | re.I)
        if not script_match:
            return None
        
        script_content = script_match.group(1)
        logger.info(f"[agent] found script: {script_content[:200]}...")
        
        # Pattern: const parts = [...]; const secret = parts.reduce((a, b) => a + b, 0) * N;
        parts_match = re.search(r'const\s+parts\s*=\s*\[([^\]]+)\]', script_content)
        reduce_match = re.search(r'reduce\s*\([^)]+\)\s*\*\s*(\d+)', script_content)
        
        if parts_match and reduce_match:
            parts_str = parts_match.group(1)
            parts = [int(x.strip()) for x in parts_str.split(',') if x.strip().isdigit()]
            multiplier = int(reduce_match.group(1))
            result = sum(parts) * multiplier
            logger.info(f"[agent] JS: parts={parts}, multiplier={multiplier}, result={result}")
            return result
        
        # Pattern: simple console.log with a number
        console_match = re.search(r'console\.log\s*\([^)]*?(\d+)[^)]*\)', script_content)
        if console_match:
            return int(console_match.group(1))
        
        # Try to find any multiplication pattern
        mult_match = re.search(r'(\d+)\s*\*\s*(\d+)', script_content)
        if mult_match:
            return int(mult_match.group(1)) * int(mult_match.group(2))
            
    except Exception as e:
        logger.warning(f"[agent] JS evaluation failed: {e}")
    
    return None


def _sum_valid_prices(data: List[Dict]) -> Optional[float]:
    """Sum all valid numeric prices from dirty data."""
    try:
        total = 0.0
        for item in data:
            price = item.get("price")
            if price is None:
                continue
            
            if isinstance(price, (int, float)):
                total += float(price)
            elif isinstance(price, str):
                # Extract numeric value, removing currency symbols etc.
                cleaned = re.sub(r'[^0-9.]', '', price)
                if cleaned and cleaned != '.':
                    try:
                        total += float(cleaned)
                    except ValueError:
                        pass
        
        # Round to avoid floating point issues
        return round(total, 2)
        
    except Exception as e:
        logger.warning(f"[agent] price sum failed: {e}")
        return None


def _calculate_gold_tier_total(api_data: List[Dict]) -> Optional[float]:
    """Calculate total value for gold tier users."""
    try:
        users = None
        products = None
        orders = None
        
        # Find the relevant datasets
        for api_item in api_data:
            url = api_item.get("url", "").lower()
            data = api_item.get("data", [])
            
            if 'users' in url:
                users = data
            elif 'products' in url:
                products = data
            elif 'orders' in url:
                orders = data
        
        if not all([users, products, orders]):
            return None
        
        # Build product price lookup
        product_prices = {}
        for p in products:
            product_prices[p.get('id')] = p.get('price', 0)
        
        # Find gold tier user IDs
        gold_user_ids = set()
        for u in users:
            if u.get('tier', '').lower() == 'gold':
                gold_user_ids.add(u.get('id'))
        
        logger.info(f"[agent] gold user IDs: {gold_user_ids}")
        
        # Calculate total from gold user orders
        total = 0.0
        for order in orders:
            if order.get('user_id') in gold_user_ids:
                for item_id in order.get('items', []):
                    price = product_prices.get(item_id, 0)
                    total += price
                    logger.info(f"[agent] order {order.get('order_id')}: {item_id} = {price}")
        
        return total
        
    except Exception as e:
        logger.warning(f"[agent] gold tier calculation failed: {e}")
        return None


def _construct_uv_command(page_text: str, email: str) -> Optional[str]:
    """Construct UV http command string."""
    try:
        # Extract URL pattern from page
        url_match = re.search(r'(https?://[^\s]+\.json\?email=)[<\s]*your email[>\s]*', page_text, re.I)
        if url_match:
            base_url = url_match.group(1)
            full_url = base_url + email
            # Build command without quotes around URL
            cmd = f'uv http get {full_url} -H "Accept: application/json"'
            return cmd
        
        # Alternative: look for explicit URL
        url_match = re.search(r'uv http get\s+(https?://[^\s]+)', page_text)
        if url_match:
            url = url_match.group(1).replace('<your email>', email).replace('%3C', '').replace('%3E', '')
            cmd = f'uv http get {url} -H "Accept: application/json"'
            return cmd
            
    except Exception as e:
        logger.warning(f"[agent] UV command construction failed: {e}")
    return None


async def _get_dominant_color(image_url: str) -> Optional[str]:
    """Get the most frequent color from an image."""
    try:
        content = await _download_file(image_url)
        if not content:
            return None
        
        from PIL import Image
        from collections import Counter
        
        img = Image.open(io.BytesIO(content))
        img = img.convert('RGB')
        pixels = list(img.getdata())
        
        # Count all pixel colors
        color_counts = Counter(pixels)
        most_common = color_counts.most_common(1)[0][0]
        
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(*most_common)
        logger.info(f"[agent] most frequent color: {hex_color} (count: {color_counts.most_common(1)[0][1]})")
        return hex_color
        
    except Exception as e:
        logger.warning(f"[agent] dominant color extraction failed: {e}")
        return None


def _normalize_csv_to_json(df: pd.DataFrame, page_text: str) -> Optional[str]:
    """Normalize CSV to JSON with snake_case keys, ISO dates, sorted by id."""
    try:
        # Rename columns to snake_case
        column_map = {}
        for col in df.columns:
            snake = re.sub(r'(?<!^)(?=[A-Z])', '_', str(col)).lower()
            snake = re.sub(r'[^a-z0-9_]', '_', snake)
            snake = re.sub(r'_+', '_', snake).strip('_')
            column_map[col] = snake
        
        df = df.rename(columns=column_map)
        
        # Ensure standard column names
        col_mapping = {
            'id': 'id', 'name': 'name', 'joined': 'joined', 'value': 'value',
            'date': 'joined', 'val': 'value'
        }
        for old, new in col_mapping.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        # Convert dates to ISO format
        if 'joined' in df.columns:
            df['joined'] = pd.to_datetime(df['joined']).dt.strftime('%Y-%m-%d')
        
        # Convert value to int
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int)
        
        # Convert id to int
        if 'id' in df.columns:
            df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
        
        # Sort by id
        if 'id' in df.columns:
            df = df.sort_values('id')
        
        # Convert to JSON string (not Python object)
        result = df.to_json(orient='records')
        return result
        
    except Exception as e:
        logger.warning(f"[agent] CSV normalization failed: {e}")
        return None


async def _count_github_files(page_text: str, collected_data: Dict, email: str) -> Optional[int]:
    """Count files in GitHub tree under a prefix."""
    try:
        # Get the gh-tree.json config
        gh_config = None
        for json_item in collected_data.get("json_data", []):
            if 'gh-tree' in json_item.get("url", ""):
                gh_config = json_item.get("data")
                break
        
        if not gh_config:
            # Try to download it
            url_match = re.search(r'(https?://[^\s]+gh-tree\.json)', page_text)
            if url_match:
                content = await _download_file(url_match.group(1))
                if content:
                    gh_config = json.loads(content.decode())
        
        if not gh_config:
            return None
        
        owner = gh_config.get('owner')
        repo = gh_config.get('repo')
        sha = gh_config.get('sha')
        path_prefix = gh_config.get('pathPrefix', '')
        
        logger.info(f"[agent] GitHub config: {owner}/{repo} sha={sha} prefix={path_prefix}")
        
        # Call GitHub API
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
            resp.raise_for_status()
            tree_data = resp.json()
        
        # Count .md files under prefix
        count = 0
        for item in tree_data.get('tree', []):
            path = item.get('path', '')
            if path.startswith(path_prefix) and path.endswith('.md'):
                count += 1
                logger.info(f"[agent] found .md file: {path}")
        
        # Calculate offset
        email_len = len(email)
        offset = email_len % 2
        result = count + offset
        
        logger.info(f"[agent] .md count={count}, email_len={email_len}, offset={offset}, result={result}")
        return result
        
    except Exception as e:
        logger.warning(f"[agent] GitHub tree counting failed: {e}")
        return None


# ============== UTILITIES ==============

def _parse_llm_answer(answer: str) -> Any:
    """Parse LLM response into appropriate type."""
    if not answer:
        return None
    
    answer = answer.strip()
    
    # Remove markdown formatting
    if answer.startswith('```'):
        answer = re.sub(r'^```[a-z]*\n?', '', answer)
        answer = re.sub(r'\n?```$', '', answer)
        answer = answer.strip()
    
    # Remove quotes if wrapped
    if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    # Try integer
    try:
        return int(answer)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(answer)
    except ValueError:
        pass
    
    # Try JSON
    try:
        parsed = json.loads(answer)
        # Unwrap {"answer": ...} pattern
        if isinstance(parsed, dict):
            if 'answer' in parsed and len(parsed) == 1:
                return parsed['answer']
            if len(parsed) == 1:
                return list(parsed.values())[0]
        return parsed
    except:
        pass
    
    return answer


async def _call_llm(prompt: str) -> Optional[str]:
    """Call LLM via aipipe.org."""
    if not AIPIPE_TOKEN:
        logger.warning("[agent] AIPIPE_TOKEN not set")
        return None

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                AIPIPE_URL,
                headers={
                    "Authorization": f"Bearer {AIPIPE_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                logger.info(f"[agent] LLM response: {content[:200]}...")
                return content
                
    except Exception as e:
        logger.exception(f"[agent] LLM call failed: {e}")
    
    return None


async def _download_file(url: str) -> Optional[bytes]:
    """Download file content."""
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            logger.info(f"[agent] downloaded {url}: {len(resp.content)} bytes")
            return resp.content
    except Exception as e:
        logger.warning(f"[agent] download failed {url}: {e}")
        return None


async def _safe_get_text(page) -> str:
    """Safely get page text."""
    try:
        return await page.evaluate("() => document.body.innerText")
    except:
        return ""


async def _safe_get_html(page) -> str:
    """Safely get page HTML."""
    try:
        return await page.content()
    except:
        return ""


async def _extract_links(page) -> List[str]:
    """Extract all links from page."""
    try:
        anchors = await page.query_selector_all("a")
        links = []
        for a in anchors:
            href = await a.get_attribute("href")
            if href:
                links.append(href)
        return links
    except:
        return []


def _decode_atob_content(html: str) -> str:
    """Decode base64 atob() content in HTML."""
    decoded = []
    patterns = [
        r'atob\(`([^`]+)`\)',
        r'atob\("([^"]+)"\)',
        r"atob\('([^']+)'\)",
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.DOTALL):
            try:
                b64 = re.sub(r'\s+', '', match.group(1))
                decoded.append(base64.b64decode(b64).decode('utf-8', errors='ignore'))
            except:
                pass
    
    return "\n".join(decoded)


def _extract_submit_url(page_text: str, page_html: str, current_url: str) -> str:
    """Extract submit URL from page content."""
    # Patterns to match submit URLs - include optional path after /submit (like /submit/1)
    patterns = [
        r'POST[^"\'<>]*?(?:to|TO)[:\s]+(https?://[^\s<>"\']+/submit/\d+)',
        r'POST[^"\'<>]*?(?:to|TO)[:\s]+(https?://[^\s<>"\']+/submit)',
        r'<code[^>]*>(https?://[^\s<>"\']+/submit/\d+)</code>',
        r'<code[^>]*>(https?://[^\s<>"\']+/submit)</code>',
        r'"(https?://[^\s<>"]+/submit/\d+)"',
        r'"(https?://[^\s<>"]+/submit)"',
        r"'(https?://[^\s<>']+/submit/\d+)'",
        r"'(https?://[^\s<>']+/submit)'",
        r'(https?://[^\s<>"\']+/submit/\d+)',
        r'(https?://[^\s<>"\']+/submit)',
    ]
    
    for pattern in patterns:
        for text in [page_html, page_text]:  # Check HTML first for more accurate extraction
            m = re.search(pattern, text, re.I)
            if m:
                url = m.group(1)
                logger.info(f"[agent] extracted submit URL: {url}")
                return url
    
    # Fallback
    parsed = urlparse(current_url)
    return f"{parsed.scheme}://{parsed.netloc}/submit"


async def _submit_answer(submit_url: str, email: str, secret: str, page_url: str, answer: Any) -> Optional[dict]:
    """Submit answer to quiz endpoint."""
    payload = {"email": email, "secret": secret, "url": page_url, "answer": answer}
    
    logger.info(f"[agent] submitting to {submit_url}: {json.dumps(payload)[:500]}")
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(submit_url, json=payload)
            return resp.json()
    except Exception as e:
        logger.exception(f"[agent] submit failed: {e}")
        return None
