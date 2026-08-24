#!/usr/bin/env python3
"""
Automated Smoke Test Suite for Author AI Server
===============================================
Runs automated health and API endpoint checks:
1. Server GET / root UI check
2. POST /api/chat endpoint check
3. POST /api/tts speech synthesis check
4. POST /api/avatar/speak & GET /api/avatar/status/<job_id> job check
5. Error handling / bad input verification (No crash test)

Usage:
  python scripts/test_server.py
  python scripts/test_server.py --url http://localhost:8000
"""

import sys
import json
import time
import argparse
import urllib.request
import urllib.error

def run_tests(base_url="http://127.0.0.1:8000"):
    print("=" * 70)
    print(f"🧪 AUTHOR AI SERVER SMOKE TEST SUITE — Target: {base_url}")
    print("=" * 70)
    
    passed = 0
    failed = 0

    def log_result(test_name, success, details=""):
        nonlocal passed, failed
        if success:
            passed += 1
            print(f"  ✅ PASS: {test_name} {details}")
        else:
            failed += 1
            print(f"  ❌ FAIL: {test_name} — {details}")

    # Test 1: Web Interface GET /
    try:
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read().decode("utf-8")
            is_valid = resp.status == 200 and "Shahbuddin Rathod Voice AI" in content
            log_result("GET / (Web UI)", is_valid, f"Status: {resp.status}")
    except Exception as e:
        log_result("GET / (Web UI)", False, str(e))

    # Test 2: POST /api/chat
    try:
        payload = json.dumps({"query": "List 3 books by Shahbuddin Rathod"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            has_keys = "response" in data and "sources" in data and "avatar_job_id" in data
            log_result("POST /api/chat", has_keys, f"Job ID: {data.get('avatar_job_id')}")
    except Exception as e:
        log_result("POST /api/chat", False, str(e))

    # Test 3: POST /api/tts
    try:
        payload = json.dumps({"text": "Testing speech output", "voice": "en-US-AriaNeural"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/tts",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            mime = resp.headers.get("Content-Type", "")
            data_bytes = resp.read()
            is_valid = resp.status == 200 and (len(data_bytes) > 0 or "json" in mime)
            log_result("POST /api/tts", is_valid, f"MIME: {mime}, Bytes: {len(data_bytes)}")
    except Exception as e:
        log_result("POST /api/tts", False, str(e))

    # Test 4: POST /api/avatar/speak & GET /api/avatar/status/<job_id>
    job_id = None
    try:
        payload = json.dumps({"text": "Hello world avatar test"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/avatar/speak",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            job_id = data.get("job_id")
            log_result("POST /api/avatar/speak", job_id is not None, f"Enqueued Job ID: {job_id}")
    except Exception as e:
        log_result("POST /api/avatar/speak", False, str(e))

    if job_id:
        time.sleep(1)
        try:
            req = urllib.request.Request(f"{base_url}/api/avatar/status/{job_id}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                has_status = "status" in data and "engine" in data
                log_result("GET /api/avatar/status/<job_id>", has_status, f"Status: {data.get('status')}, Engine: {data.get('engine')}")
        except Exception as e:
            log_result("GET /api/avatar/status/<job_id>", False, str(e))

    # Test 5: Bad Input Error Handling (Empty Query)
    try:
        payload = json.dumps({"query": ""}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                log_result("Error Handling (Empty Query)", False, f"Unexpected status {resp.status}")
        except urllib.error.HTTPError as err:
            is_handled = err.code == 400
            log_result("Error Handling (Empty Query)", is_handled, f"Returned HTTP {err.code} as expected")
    except Exception as e:
        log_result("Error Handling (Empty Query)", False, str(e))

    print("=" * 70)
    print(f"📊 TEST SUMMARY: {passed} PASSED | {failed} FAILED")
    print("=" * 70)
    return failed == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Author AI Server Test Runner")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of server")
    args = parser.parse_args()
    
    success = run_tests(args.url)
    sys.exit(0 if success else 1)
