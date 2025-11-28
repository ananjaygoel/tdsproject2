# agent.py
"""
LLM-powered quiz solver agent using aipipe.org (OpenAI-compatible API).
Visits quiz URLs, uses LLM to understand tasks, executes actions, and submits answers.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Optional, Any
from urllib.parse import urljoin, urlparse

import httpx
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from dotenv import load_dotenv

load_dotenv()

# ------- Configuration -------
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-nano")  # Fast model via aipipe
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
        # No running loop -> run in a new thread with its own event loop
        def _target():
            try:
                asyncio.run(_run_agent(start_url, email, secret))
            except Exception:
                logger.exception("[agent] run in thread crashed")

        t = threading.Thread(target=_target, daemon=True)
        t.start()
    else:
        # There's an event loop running (FastAPI uvicorn). Schedule a task.
        asyncio.create_task(_run_agent(start_url, email, secret))


async def _run_agent(start_url: str, email: Optional[str], secret: Optional[str]):
    """
    Async worker — visits the quiz URL, uses LLM to solve tasks, submits answers,
    and follows the next URL until done or time runs out.
    """
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
                now = time.monotonic()
                if now > deadline:
                    logger.warning("[agent] reached time limit, stopping")
                    break

                logger.info(f"[agent] visiting {current_url}")
                
                # Load the page
                try:
                    await page.goto(current_url, timeout=60000, wait_until="networkidle")
                except PlaywrightTimeoutError:
                    logger.warning("[agent] page.goto timed out; trying domcontentloaded")
                    try:
                        await page.goto(current_url, timeout=30000, wait_until="domcontentloaded")
                    except Exception as e:
                        logger.exception(f"[agent] page.goto final fail: {e}")
                        break
                except Exception:
                    logger.exception("[agent] page.goto unexpected error")
                    break

                # Wait a bit for JS to execute
                await asyncio.sleep(1)

                # Get rendered content
                try:
                    page_text = await page.evaluate("() => document.body.innerText")
                except Exception:
                    page_text = ""
                try:
                    page_html = await page.content()
                except Exception:
                    page_html = ""

                logger.info(f"[agent] page text preview: {page_text[:500]}...")

                # Extract submit URL from the page (don't hardcode!)
                submit_url = _extract_submit_url(page_text, page_html, current_url)
                logger.info(f"[agent] extracted submit URL: {submit_url}")

                # Extract any links from the page
                links = await _extract_links(page)
                
                # Solve the task using LLM
                answer = await _solve_task(
                    current_url=current_url,
                    page_text=page_text,
                    page_html=page_html,
                    links=links,
                    email=email,
                    secret=secret
                )
                
                if answer is None:
                    logger.warning("[agent] could not determine answer, stopping")
                    break

                logger.info(f"[agent] submitting answer: {str(answer)[:200]}")

                # Submit the answer
                resp = await _submit_answer(
                    submit_url=submit_url,
                    email=email,
                    secret=secret,
                    page_url=current_url,
                    answer=answer
                )

                if not resp:
                    logger.error("[agent] submit returned no response")
                    break

                logger.info(f"[agent] submit response: {resp}")

                if resp.get("correct") is True:
                    logger.info("[agent] answer correct!")
                    next_url = resp.get("url")
                    if next_url:
                        current_url = next_url
                        continue
                    else:
                        logger.info("[agent] no more URLs, quiz complete!")
                        break
                else:
                    reason = resp.get("reason", "unknown")
                    logger.warning(f"[agent] answer incorrect: {reason}")
                    # Check if there's a next URL to continue anyway
                    next_url = resp.get("url")
                    if next_url:
                        logger.info(f"[agent] moving to next URL despite incorrect answer")
                        current_url = next_url
                        continue
                    else:
                        break

        except Exception:
            logger.exception("[agent] unexpected error in run loop")
        finally:
            try:
                await context.close()
                await browser.close()
            except Exception:
                pass

    logger.info("[agent] finished worker")


def _extract_submit_url(page_text: str, page_html: str, current_url: str) -> str:
    """
    Extract the submit URL from the page content.
    The quiz always includes the submit URL - we should NOT hardcode it.
    """
    # Look for common patterns
    patterns = [
        r'POST[^"\'<>]*?(?:to|TO)\s+(https?://[^\s<>"\']+/submit)',
        r'post[^"\'<>]*?(?:to|TO)\s+(https?://[^\s<>"\']+/submit)',
        r'"(https?://[^\s<>"]+/submit)"',
        r"'(https?://[^\s<>']+/submit)'",
        r'(https?://[^\s<>"\']+/submit)',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, page_text, re.I)
        if m:
            return m.group(1)
        m = re.search(pattern, page_html, re.I)
        if m:
            return m.group(1)
    
    # Fallback: construct from current URL's origin
    parsed = urlparse(current_url)
    return f"{parsed.scheme}://{parsed.netloc}/submit"


async def _solve_task(
    current_url: str,
    page_text: str,
    page_html: str,
    links: list,
    email: str,
    secret: str
) -> Optional[Any]:
    """
    Main task solver. Uses pattern matching first, then LLM for complex tasks.
    """
    
    # Decode any base64 content (atob patterns) first
    decoded_content = _decode_atob_content(page_html)
    full_text = page_text
    if decoded_content:
        full_text = page_text + "\n\n[DECODED CONTENT]:\n" + decoded_content
        logger.info(f"[agent] decoded base64 content: {decoded_content[:300]}...")

    # === PATTERN MATCHING (fast path) ===
    
    # 1. Check for demo page (accepts any answer)
    if "anything you want" in full_text.lower():
        logger.info("[agent] detected demo page, submitting 'demo'")
        return "demo"

    # 2. Check for scrape-data endpoint pattern (e.g., "Scrape /demo-scrape-data?...")
    scrape_match = re.search(r'[Ss]crape\s+(/[^\s]+)', full_text)
    if scrape_match:
        scrape_path = scrape_match.group(1)
        scrape_url = urljoin(current_url, scrape_path)
        logger.info(f"[agent] found scrape task, fetching: {scrape_url}")
        
        scraped_data = await _download_file(scrape_url)
        if scraped_data:
            scraped_text = scraped_data.decode('utf-8', errors='ignore')
            logger.info(f"[agent] scraped content: {scraped_text[:500]}...")
            
            # Look for secret code in scraped content (alphanumeric)
            secret_in_scraped = re.search(r'[Ss]ecret\s*[Cc]ode[:\s]+([a-zA-Z0-9]+)', scraped_text)
            if secret_in_scraped:
                answer = secret_in_scraped.group(1)
                logger.info(f"[agent] found secret code in scraped data: {answer}")
                return answer
            
            # Also check for just a code/hash pattern
            code_match = re.search(r'\b([a-f0-9]{6,})\b', scraped_text, re.I)
            if code_match:
                answer = code_match.group(1)
                logger.info(f"[agent] found code in scraped data: {answer}")
                return answer

    # 3. Check for simple "secret code" pattern
    secret_match = re.search(r"[Ss]ecret\s*[Cc]ode\s*(?:is|=|:)\s*([a-zA-Z0-9]+)", full_text)
    if secret_match:
        logger.info(f"[agent] found secret code pattern: {secret_match.group(1)}")
        return secret_match.group(1)

    # 4. CSV with cutoff task
    cutoff_match = re.search(r'[Cc]utoff[:\s]*(\d+)', full_text)
    csv_links = [l for l in links if l and '.csv' in l.lower()]
    # Also search for CSV URLs in HTML
    csv_url_match = re.search(r'(https?://[^\s<>"\']+\.csv)', page_html, re.I)
    if csv_url_match and not csv_links:
        csv_links.append(csv_url_match.group(1))
    
    if csv_links:
        csv_url = csv_links[0] if csv_links[0].startswith('http') else urljoin(current_url, csv_links[0])
        cutoff = int(cutoff_match.group(1)) if cutoff_match else 0
        logger.info(f"[agent] found CSV task: {csv_url}, cutoff={cutoff}")
        
        result = await _process_csv_task(csv_url, cutoff, full_text)
        if result is not None:
            return result

    # 5. Check for JSON data tasks
    json_links = [l for l in links if l and '.json' in l.lower()]
    if json_links:
        json_url = json_links[0] if json_links[0].startswith('http') else urljoin(current_url, json_links[0])
        logger.info(f"[agent] found JSON link: {json_url}")
        result = await _process_json_task(json_url, full_text)
        if result is not None:
            return result

    # === LLM PATH (for complex tasks) ===
    answer = await _solve_with_llm(
        current_url=current_url,
        page_text=full_text,
        page_html=page_html,
        links=links
    )
    
    if answer is not None:
        return answer

    # === FALLBACK: extract any obvious answer ===
    # Look for a single prominent number
    numbers = re.findall(r'\b(\d{2,10})\b', full_text)
    if len(numbers) == 1:
        logger.info(f"[agent] fallback: single number found: {numbers[0]}")
        return int(numbers[0])

    return None


def _decode_atob_content(html: str) -> str:
    """Decode any atob() base64 content in the HTML."""
    decoded_parts = []
    
    # Find atob patterns (handles backticks, single and double quotes)
    patterns = [
        r'atob\(`([^`]+)`\)',
        r'atob\("([^"]+)"\)',
        r"atob\('([^']+)'\)",
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.DOTALL):
            try:
                b64 = match.group(1)
                # Handle multi-line base64 (remove whitespace)
                b64_clean = re.sub(r'\s+', '', b64)
                decoded = base64.b64decode(b64_clean).decode('utf-8', errors='ignore')
                decoded_parts.append(decoded)
            except Exception as e:
                logger.warning(f"[agent] failed to decode base64: {e}")
                continue
    
    return "\n".join(decoded_parts)


async def _process_csv_task(csv_url: str, cutoff: int, task_text: str) -> Optional[int]:
    """Download and process CSV for sum/count/filter tasks."""
    try:
        csv_data = await _download_file(csv_url)
        if not csv_data:
            return None
        
        df = pd.read_csv(io.BytesIO(csv_data))
        logger.info(f"[agent] CSV loaded: {df.shape}, columns: {list(df.columns)}")
        
        # Determine what operation to perform based on task text
        task_lower = task_text.lower()
        
        # Find numeric column(s)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Check for specific column mentions
        target_col = None
        for col in df.columns:
            if col.lower() in task_lower or col.lower().replace('_', ' ') in task_lower:
                target_col = col
                break
        
        if target_col is None and numeric_cols:
            target_col = numeric_cols[0]
        
        if target_col is None:
            # Use first column
            target_col = df.columns[0]
        
        logger.info(f"[agent] using column: {target_col}")
        
        # Convert to numeric
        values = pd.to_numeric(df[target_col], errors='coerce').dropna()
        
        # Apply cutoff filter if specified
        if cutoff > 0:
            values = values[values >= cutoff]
            logger.info(f"[agent] after cutoff filter >= {cutoff}: {len(values)} values")
        
        # Determine operation
        if 'count' in task_lower or 'how many' in task_lower:
            result = len(values)
        elif 'average' in task_lower or 'mean' in task_lower:
            result = values.mean()
        elif 'max' in task_lower or 'maximum' in task_lower or 'largest' in task_lower:
            result = values.max()
        elif 'min' in task_lower or 'minimum' in task_lower or 'smallest' in task_lower:
            result = values.min()
        else:
            # Default to sum
            result = values.sum()
        
        logger.info(f"[agent] CSV result: {result}")
        return int(result) if pd.notna(result) else None
        
    except Exception as e:
        logger.exception(f"[agent] CSV processing failed: {e}")
        return None


async def _process_json_task(json_url: str, task_text: str) -> Optional[Any]:
    """Download and process JSON data."""
    try:
        data = await _download_file(json_url)
        if not data:
            return None
        
        json_data = json.loads(data.decode('utf-8'))
        logger.info(f"[agent] JSON loaded: {type(json_data)}")
        
        # Use LLM to process JSON
        prompt = f"""Analyze this JSON data and answer the question.

QUESTION/TASK:
{task_text[:2000]}

JSON DATA:
{json.dumps(json_data, indent=2)[:3000]}

Return ONLY the answer value (number, string, or JSON). No explanation."""

        return await _call_llm(prompt)
        
    except Exception as e:
        logger.exception(f"[agent] JSON processing failed: {e}")
        return None


async def _solve_with_llm(
    current_url: str,
    page_text: str,
    page_html: str,
    links: list
) -> Optional[Any]:
    """Use LLM to understand the task and determine the answer."""
    
    if not AIPIPE_TOKEN:
        logger.warning("[agent] AIPIPE_TOKEN not set, skipping LLM")
        return None

    # Build context
    link_text = "\n".join([f"- {link}" for link in links[:15]]) if links else "No links found"
    
    prompt = f"""You are solving a data analysis quiz. Read the task and provide the answer.

URL: {current_url}

PAGE CONTENT:
{page_text[:4000]}

LINKS:
{link_text}

INSTRUCTIONS:
1. Read the task carefully
2. If it's a calculation (sum, count, filter), compute it
3. If it mentions a secret code, extract it
4. Return ONLY the raw answer value

IMPORTANT: Return ONLY the answer itself. Examples:
- If secret code is "abc123" → respond: abc123
- If sum is 42 → respond: 42
- DO NOT wrap in JSON like {{"answer": ...}}
- DO NOT add explanation or quotes

RESPOND WITH ONLY THE RAW ANSWER VALUE."""

    try:
        answer = await _call_llm(prompt)
        if answer:
            answer = answer.strip()
            # Remove markdown code fences if present
            if answer.startswith('```'):
                answer = re.sub(r'^```[a-z]*\n?', '', answer)
                answer = re.sub(r'\n?```$', '', answer)
                answer = answer.strip()
            
            # Try to parse as appropriate type
            try:
                # Try integer first
                return int(answer)
            except ValueError:
                try:
                    # Try float
                    return float(answer)
                except ValueError:
                    try:
                        # Try JSON - but extract the value if it's a simple {"answer": ...} wrapper
                        parsed = json.loads(answer)
                        if isinstance(parsed, dict):
                            # If it's a dict with 'answer' key, extract that value
                            if 'answer' in parsed and len(parsed) == 1:
                                return parsed['answer']
                            # If it has a single key, return that value
                            if len(parsed) == 1:
                                return list(parsed.values())[0]
                        return parsed
                    except:
                        # Return as string
                        return answer
    except Exception as e:
        logger.exception(f"[agent] LLM solving failed: {e}")
    
    return None


async def _download_file(url: str) -> Optional[bytes]:
    """Download a file and return its content."""
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            logger.info(f"[agent] downloaded {url}: {len(resp.content)} bytes")
            return resp.content
    except Exception as e:
        logger.warning(f"[agent] failed to download {url}: {e}")
        return None


async def _call_llm(prompt: str) -> Optional[str]:
    """Call the LLM via aipipe.org."""
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
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                logger.info(f"[agent] LLM response: {content[:200]}...")
                return content
            
            logger.warning(f"[agent] unexpected LLM response: {data}")
            return None
    except Exception as e:
        logger.exception(f"[agent] LLM API call failed: {e}")
        return None


async def _submit_answer(
    submit_url: str,
    email: str,
    secret: str,
    page_url: str,
    answer: Any
) -> Optional[dict]:
    """Submit answer to the quiz endpoint."""
    payload = {
        "email": email,
        "secret": secret,
        "url": page_url,
        "answer": answer
    }
    
    logger.info(f"[agent] POSTing to {submit_url}: {json.dumps(payload)[:500]}")
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(submit_url, json=payload)
            try:
                return resp.json()
            except:
                logger.warning(f"[agent] submit response not JSON: {resp.text[:200]}")
                return None
    except Exception as e:
        logger.exception(f"[agent] submit request failed: {e}")
        return None


async def _extract_links(page) -> list:
    """Extract all href links from the page."""
    hrefs = []
    try:
        anchors = await page.query_selector_all("a")
        for a in anchors:
            h = await a.get_attribute("href")
            if h:
                hrefs.append(h)
    except Exception:
        logger.exception("[agent] extracting links failed")
    return hrefs
