# agent.py
import asyncio
import base64
import io
import logging
import re
import threading
import time
from typing import Optional
from urllib.parse import urljoin

import httpx
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ------- Configuration -------
SUBMIT_ENDPOINT = "https://tds-llm-analysis.s-anand.net/submit"
TIME_LIMIT_SECONDS = 180  # total seconds allowed per quiz run
# Default values (main.py passes actual email/secret)
DEFAULT_EMAIL = None
DEFAULT_SECRET = None
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")


def run_agent(start_url: str, email: Optional[str] = None, secret: Optional[str] = None):
    """
    Entrypoint expected by FastAPI BackgroundTasks.
    Schedules the async worker safely whether called from an event loop or not.
    """
    # allow passing defaults if main.py didn't include (but main does)
    email = email or DEFAULT_EMAIL
    secret = secret or DEFAULT_SECRET

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop -> run in a new thread with its own event loop to avoid blocking FastAPI
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
    Async worker — visits the quiz URL, solves tasks, submits answers, and follows the next URL until done
    or time runs out.
    """
    logger.info(f"[agent] starting worker for url={start_url} email={email}")

    start_time = time.monotonic()
    deadline = start_time + TIME_LIMIT_SECONDS
    current_url = start_url

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()

        try:
            while current_url:
                now = time.monotonic()
                if now > deadline:
                    logger.warning("[agent] reached time limit, stopping")
                    break

                logger.info(f"[agent] visiting {current_url}")
                try:
                    await page.goto(current_url, timeout=60000, wait_until="domcontentloaded")
                except PlaywrightTimeoutError:
                    logger.warning("[agent] page.goto timed out; trying networkidle then continuing")
                    try:
                        await page.goto(current_url, timeout=120000, wait_until="networkidle")
                    except Exception as e:
                        logger.exception(f"[agent] page.goto final fail: {e}")
                        break
                except Exception:
                    logger.exception("[agent] page.goto unexpected error")
                    break

                # Grab rendered text and full HTML (await properly!)
                try:
                    page_text = await page.evaluate("() => document.documentElement.innerText")
                except Exception:
                    page_text = ""
                try:
                    page_html = await page.content()
                except Exception:
                    page_html = page_text or ""

                # Try handlers in order
                # 1) Look for explicit 'Secret code is NNN' in visible text
                m = re.search(r"Secret code\s*(?:is|=|:)\s*([0-9]+)", page_text, re.I)
                if m:
                    answer = m.group(1)
                    logger.info(f"[agent] found secret in page text: {answer}")
                    resp = await submit_answer(email, secret, current_url, answer)
                    if not await _process_submit_response(resp):
                        break
                    current_url = resp.get("url")
                    continue

                # 2) Attempt to decode any atob(...) base64 payloads in HTML (JS-embedded)
                atob_b64s = re.findall(r'atob\((?:`([^`]+)`|"([^"]+)"|\'([^\']+)\')\)', page_html)
                # atob_b64s is list of tuples; flatten
                b64s = []
                for t in atob_b64s:
                    for s in t:
                        if s:
                            b64s.append(s)
                decoded_texts = []
                for b in b64s:
                    try:
                        decoded_texts.append(base64.b64decode(b).decode("utf-8", errors="ignore"))
                    except Exception:
                        continue

                # Search decoded_texts for secret or instructions
                found_in_decoded = False
                for decoded in decoded_texts:
                    m2 = re.search(r"Secret code\s*(?:is|=|:)\s*([0-9]+)", decoded, re.I)
                    if m2:
                        answer = m2.group(1)
                        logger.info("[agent] found secret in decoded atob payload")
                        resp = await submit_answer(email, secret, current_url, answer)
                        if not await _process_submit_response(resp):
                            return
                        current_url = resp.get("url")
                        found_in_decoded = True
                        break
                if found_in_decoded:
                    continue

                # 3) If page mentions 'demo-scrape-data' or similar, try to fetch that endpoint (httpx)
                links = await _extract_links(page)
                demo_data_link = None
                for href in links:
                    if href and "demo-scrape-data" in href:
                        demo_data_link = urljoin(current_url, href)
                        break

                if demo_data_link:
                    try:
                        async with httpx.AsyncClient(timeout=30) as client:
                            r = await client.get(demo_data_link)
                            txt = r.text
                        m3 = re.search(r"Secret code\s*(?:is|=|:)\s*([0-9]+)", txt, re.I)
                        if m3:
                            answer = m3.group(1)
                            logger.info("[agent] found secret in demo-scrape-data endpoint")
                            resp = await submit_answer(email, secret, current_url, answer)
                            if not await _process_submit_response(resp):
                                return
                            current_url = resp.get("url")
                            continue
                    except Exception:
                        logger.exception("[agent] fetching demo-scrape-data failed")

                # 4) CSV tasks: if the page mentions 'Cutoff' or 'download this csv', find CSV link and sum values >= cutoff
                if re.search(r"Cutoff[:\s]|download.*csv|download this csv", page_text, re.I):
                    csv_link = None
                    for href in links:
                        if href and href.lower().endswith(".csv"):
                            csv_link = urljoin(current_url, href)
                            break
                    # also look for CSV URL inside HTML
                    if not csv_link:
                        m_csv = re.search(r"https?://[^\s'\"]+\.csv", page_html, re.I)
                        if m_csv:
                            csv_link = m_csv.group(0)
                    if csv_link:
                        logger.info(f"[agent] found CSV link: {csv_link}")
                        # find cutoff
                        cm = re.search(r"Cutoff[:\s]*([0-9]+)", page_text, re.I)
                        cutoff = int(cm.group(1)) if cm else 0
                        try:
                            async with httpx.AsyncClient(timeout=60) as client:
                                r = await client.get(csv_link)
                                r.raise_for_status()
                                csv_bytes = r.content
                            s = await asyncio.to_thread(_sum_csv_first_column, csv_bytes, cutoff)
                            logger.info(f"[agent] computed CSV sum >= {cutoff} => {s}")
                            resp = await submit_answer(email, secret, current_url, int(s))
                            if not await _process_submit_response(resp):
                                return
                            current_url = resp.get("url")
                            continue
                        except Exception:
                            logger.exception("[agent] csv download or processing failed")

                # 5) Last resort: find numbers in visible text and try a heuristic (common fallback)
                nums = [int(x) for x in re.findall(r"\b([0-9]{2,12})\b", page_text)]
                if nums:
                    # If exactly one obvious number -> submit it; otherwise submit sum
                    answer = nums[0] if len(nums) == 1 else sum(nums)
                    logger.info(f"[agent] fallback numeric answer -> {answer}")
                    resp = await submit_answer(email, secret, current_url, answer)
                    if not await _process_submit_response(resp):
                        return
                    current_url = resp.get("url")
                    continue

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


async def _process_submit_response(resp_json: Optional[dict]) -> bool:
    """
    Return True to continue to next URL (if any), False to stop.
    """
    if not resp_json:
        logger.error("[agent] submit returned no JSON")
        return False
    if resp_json.get("correct") is True:
        logger.info("[agent] submit correct")
        return True
    else:
        reason = resp_json.get("reason", "<no reason>")
        logger.warning(f"[agent] submit incorrect or failed: {reason}")
        return False


async def submit_answer(email: Optional[str], secret: Optional[str], page_url: str, answer):
    """
    Post answer to the judge endpoint. Returns parsed JSON or None.
    """
    payload = {"email": email, "secret": secret, "url": page_url, "answer": answer}
    logger.info(f"[agent] submitting answer for {page_url} -> {str(answer)[:200]}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(SUBMIT_ENDPOINT, json=payload)
            try:
                j = r.json()
            except Exception:
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


def _sum_csv_first_column(csv_bytes: bytes, cutoff: int) -> int:
    """
    Parse CSV bytes into pandas DataFrame, take first column, sum values >= cutoff.
    Handles file with/without header robustly.
    """
    try:
        buf = io.BytesIO(csv_bytes)
        # try without headers first
        try:
            df = pd.read_csv(buf, header=None)
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf, header=0)
        if df.shape[1] < 1:
            raise ValueError("CSV has no columns")
        col = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(int)
        col = col[col >= cutoff]
        return int(col.sum())
    except Exception:
        logger.exception("[agent] csv parse failed; falling back to manual parse")
        s = 0
        try:
            text = csv_bytes.decode("utf-8", errors="ignore")
            for line in text.splitlines():
                m = re.findall(r"-?\d+", line)
                if m:
                    v = int(m[0])
                    if v >= cutoff:
                        s += v
            return s
        except Exception:
            return 0


async def _extract_links(page) -> list:
    """
    Return absolute/relative hrefs from anchor tags (raw strings).
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
