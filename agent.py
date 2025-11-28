# agent.py
import asyncio
import time
import logging
import re
import os
import base64
import io
from urllib.parse import urljoin, urlparse
import httpx
import pandas as pd

from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

SUBMIT_ENDPOINT = "https://tds-llm-analysis.s-anand.net/submit"
TIME_LIMIT_SECONDS = 180  # 3 minutes


def run_agent(url: str, email: str, secret: str):
    """
    Synchronous wrapper expected by FastAPI background tasks.
    Schedules the async worker and returns immediately.
    """
    try:
        asyncio.create_task(_run_agent(url, email, secret))
    except RuntimeError:
        # If event loop is not running, start a task on a new loop in thread
        loop = asyncio.new_event_loop()
        asyncio.ensure_future(_run_agent(url, email, secret), loop=loop)
        # run loop in background thread
        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()
        import threading
        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()


async def _run_agent(start_url: str, email: str, secret: str):
    """
    The main async worker that performs the multi-step quiz solving loop.
    """
    start_time = time.monotonic()
    deadline = start_time + TIME_LIMIT_SECONDS
    current_url = start_url

    logger.info(f"[agent] starting worker for {email} target={current_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()

        try:
            while True:
                now = time.monotonic()
                if now > deadline:
                    logger.warning("[agent] time limit exceeded, stopping")
                    break

                logger.info(f"[agent] visiting {current_url}")
                try:
                    await page.goto(current_url, timeout=60000)
                except Exception as e:
                    logger.exception(f"[agent] page.goto failed: {e}")
                    # If goto fails, try fetching via httpx as fallback
                    try:
                        async with httpx.AsyncClient(timeout=30) as client:
                            r = await client.get(current_url)
                            page_content_text = r.text
                    except Exception as e2:
                        logger.exception(f"[agent] fallback fetch failed: {e2}")
                        break
                else:
                    # Prefer rendered visible text
                    try:
                        page_content_text = await page.evaluate("() => document.documentElement.innerText")
                    except Exception:
                        page_content_text = await page.content()

                # Try to find easy direct answers in visible text
                # 1) Secret code patterns like: "Secret code is 14940"
                m = re.search(r"Secret code\s*(?:is|=|:)\s*([0-9]+)", page_content_text, re.I)
                if m:
                    answer = m.group(1)
                    logger.info(f"[agent] found secret code in page text: {answer}")
                    resp = await submit_answer(email, secret, current_url, answer)
                    if handle_submit_response(resp):
                        current_url = resp.get("url") or None
                        if not current_url:
                            break
                        continue
                    else:
                        break

                # 2) If page instructs to scrape /demo-scrape-data or similar, attempt to fetch that data endpoint
                # Look for any path like /demo-scrape-data or links with href containing 'demo-scrape-data'
                hrefs = await _extract_links_from_page(page)
                data_href = None
                for h in hrefs:
                    if h and ("demo-scrape-data" in h or h.endswith(".csv") or h.endswith(".pdf")):
                        data_href = urljoin(current_url, h)
                        break

                # If there's a demo-scrape-data endpoint visible, fetch and look for secret
                if data_href and "demo-scrape-data" in data_href:
                    logger.info(f"[agent] found demo-scrape-data link: {data_href}")
                    try:
                        async with httpx.AsyncClient(timeout=30) as client:
                            r = await client.get(data_href)
                            txt = r.text
                        m2 = re.search(r"Secret code\s*(?:is|=|:)\s*([0-9]+)", txt, re.I)
                        if m2:
                            answer = m2.group(1)
                            logger.info(f"[agent] found secret from data endpoint: {answer}")
                            resp = await submit_answer(email, secret, current_url, answer)
                            if handle_submit_response(resp):
                                current_url = resp.get("url") or None
                                if not current_url:
                                    break
                                continue
                            else:
                                break
                    except Exception:
                        logger.exception("[agent] fetching demo-scrape-data failed")

                # 3) If page contains CSV download instructions (audio task)
                # Look for keywords that indicate CSV + cutoff instruction
                if re.search(r"download.*csv|Cutoff:|cutoff", page_content_text, re.I):
                    # find CSV link among hrefs
                    csv_link = None
                    for h in hrefs:
                        if h and h.lower().endswith(".csv"):
                            csv_link = urljoin(current_url, h)
                            break
                    # also accept links containing '/demo-audio' data attachments
                    if not csv_link:
                        for h in hrefs:
                            if h and "download" in h.lower() and ".csv" in h.lower():
                                csv_link = urljoin(current_url, h)
                                break
                    if csv_link:
                        logger.info(f"[agent] downloading csv: {csv_link}")
                        try:
                            async with httpx.AsyncClient(timeout=60) as client:
                                r = await client.get(csv_link)
                                r.raise_for_status()
                                content = r.content
                            # parse cutoff
                            cm = re.search(r"Cutoff[:\s]*([0-9]+)", page_content_text, re.I)
                            cutoff = None
                            if cm:
                                cutoff = int(cm.group(1))
                            else:
                                # fallback: try to find "cutoff" in linked filename or page
                                logger.info("[agent] cutoff not found in page text; defaulting to 0")
                                cutoff = 0
                            s = compute_sum_from_csv_bytes(content, cutoff)
                            logger.info(f"[agent] computed sum >= {cutoff} -> {s}")
                            resp = await submit_answer(email, secret, current_url, int(s))
                            if handle_submit_response(resp):
                                current_url = resp.get("url") or None
                                if not current_url:
                                    break
                                continue
                            else:
                                break
                        except Exception:
                            logger.exception("[agent] csv download/processing failed")

                # 4) If page contains base64 encoded task embedded in script (sample pages)
                # Look for atob(`...`) or atob("...") patterns
                atob_matches = re.findall(r"atob\(`([\s\S]+?)`\)|atob\(\"([^\"]+)\"\)|atob\('([^']+)'\)", page.content() if hasattr(page, "content") else page_content_text)
                # flatten matches
                for tup in atob_matches:
                    b64 = next((x for x in tup if x), None)
                    if b64:
                        try:
                            decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
                            m3 = re.search(r"Secret code\s*(?:is|=|:)\s*([0-9]+)", decoded, re.I)
                            if m3:
                                answer = m3.group(1)
                                logger.info("[agent] found secret in base64 script")
                                resp = await submit_answer(email, secret, current_url, answer)
                                if handle_submit_response(resp):
                                    current_url = resp.get("url") or None
                                    if not current_url:
                                        break
                                    continue
                                else:
                                    break
                        except Exception:
                            continue

                # 5) Last resort: attempt to let a simple heuristic compute numbers (sum of numbers on page)
                nums = [int(x) for x in re.findall(r"\b([0-9]{2,12})\b", page_content_text)]
                if nums:
                    # If there is one obvious small integer (like 12345) prefer that
                    # Otherwise submit the sum as a fallback
                    if len(nums) == 1:
                        answer = str(nums[0])
                    else:
                        answer = str(sum(nums))
                    logger.info(f"[agent] fallback answer from page numbers: {answer}")
                    resp = await submit_answer(email, secret, current_url, answer)
                    if handle_submit_response(resp):
                        current_url = resp.get("url") or None
                        if not current_url:
                            break
                        continue
                    else:
                        break

                # If nothing matched, abort
                logger.warning("[agent] no handler matched this page; stopping")
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


def handle_submit_response(resp_json):
    """
    Returns True if we should continue to next URL, False to stop.
    Side-effect: logs message.
    """
    if not resp_json:
        logger.error("[agent] no response from submit")
        return False
    if resp_json.get("correct") is True:
        logger.info("[agent] answer correct")
        return True
    else:
        reason = resp_json.get("reason") or "<no reason>"
        logger.warning(f"[agent] submission incorrect or failed: {reason}")
        return False


async def submit_answer(email, secret, page_url, answer):
    """
    Posts the answer to the grading server.
    The 'url' field should be the quiz page URL the agent solved (full URL).
    Returns parsed JSON or None on error.
    """
    payload = {
        "email": email,
        "secret": secret,
        "url": page_url,
        "answer": answer
    }
    logger.info(f"[agent] submitting answer payload for {page_url} -> {str(answer)[:200]}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(SUBMIT_ENDPOINT, json=payload)
            try:
                j = r.json()
            except Exception:
                logger.exception("[agent] submit returned non-json")
                return None
            logger.info(f"[agent] submit response: {j}")
            return j
    except Exception:
        logger.exception("[agent] submit request failed")
        return None


async def _extract_links_from_page(page):
    """
    Return list of hrefs found in <a> elements on the page.
    """
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


def compute_sum_from_csv_bytes(content_bytes: bytes, cutoff: int) -> int:
    """
    Read CSV bytes into pandas, take first column, sum values >= cutoff.
    Assumes no header (handles both header/no-header).
    """
    try:
        # Try with header inference off first
        buf = io.BytesIO(content_bytes)
        df = None
        try:
            df = pd.read_csv(buf, header=None)
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf, header=0)
        # ensure at least one column
        if df.shape[1] < 1:
            raise ValueError("CSV has no columns")
        col = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(int)
        col = col[col >= cutoff]
        return int(col.sum())
    except Exception:
        logger.exception("[agent] compute_sum_from_csv_bytes failed")
        # fallback: attempt manual parse
        s = 0
        try:
            txt = content_bytes.decode("utf-8", errors="ignore").strip().splitlines()
            for r in txt:
                val = re.findall(r"-?\d+", r)
                if val:
                    v = int(val[0])
                    if v >= cutoff:
                        s += v
            return s
        except Exception:
            return 0
