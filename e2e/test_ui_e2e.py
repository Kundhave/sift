"""Playwright end-to-end tests. Starts the CPU-only backend (make backend) and the built frontend
(make demo's preview server) as subprocesses. Run: .venv/bin/python -m pytest e2e -q"""
import os, socket, subprocess, time, urllib.request
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
API, UI = 8765, 5175
SHOTS = ROOT / "reports" / "screenshots"


def _up(url, t=90):
    end = time.time() + t
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2); return True
        except Exception:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def servers():
    env = dict(os.environ, SIFT_API=f"http://127.0.0.1:{API}", CUDA_VISIBLE_DEVICES="", SIFT_DEVICE="cpu", PYTHONPATH=str(ROOT / "src"))
    procs = []
    # backend via the make target's command; ports overridden. Frontend proxies /api? check vite config.
    procs.append(subprocess.Popen(["make", "backend", f"API_PORT={API}"], cwd=ROOT, env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
    procs.append(subprocess.Popen(["npm", "--prefix", "app/frontend", "run", "preview", "--", "--port", str(UI), "--strictPort"],
                                  cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
    assert _up(f"http://127.0.0.1:{API}/api/health"), "backend did not start"
    assert _up(f"http://127.0.0.1:{UI}/"), "frontend did not start"
    yield f"http://127.0.0.1:{UI}"
    import signal
    for p in procs:
        try: os.killpg(p.pid, signal.SIGTERM)
        except Exception: pass


@pytest.fixture(scope="session")
def browser_ctx():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture()
def page(browser_ctx, servers):
    ctx = browser_ctx.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    yield pg
    ctx.close()


def test_review_sample_analyse_and_cards(page, servers):
    page.goto(servers + "/app")
    sel = page.locator("#sample")
    page.wait_for_function("document.querySelectorAll('#sample option').length > 1", timeout=30000)
    sel.select_option(index=1)
    page.get_by_role("button", name="Analyse").click()
    page.get_by_test_id("summary").wait_for(timeout=180000)
    cards = page.locator("article[aria-label^='Hunk ']")
    cards.first.wait_for(timeout=30000)
    assert cards.count() >= 1
    assert "misaligned" in cards.first.get_attribute("aria-label")
    page.get_by_role("heading", name="Consolidated intent").wait_for()
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / "e2e_review_desktop.png"))
    assert not page.errors


def test_model_switch_reanalyses(page, servers):
    page.goto(servers + "/app")
    page.wait_for_function("document.querySelectorAll('#sample option').length > 1", timeout=30000)
    page.locator("#sample").select_option(index=1)
    page.get_by_role("button", name="Analyse").click()
    page.get_by_test_id("summary").wait_for(timeout=180000)
    opts = page.locator("#model option:not([disabled])")
    n = opts.count()
    assert n >= 1
    if n < 2:
        pytest.skip("only one model available")
    values = [opts.nth(i).get_attribute("value") for i in range(n)]
    cur = page.locator("#model").input_value()
    other = next(v for v in values if v != cur)
    with page.expect_response(lambda r: r.url.endswith("/api/analyze") and r.status == 200, timeout=180000):
        page.locator("#model").select_option(other)
    page.get_by_test_id("summary").wait_for()
    assert page.locator("#model").input_value() == other


def test_custom_input_analyse(page, servers):
    page.goto(servers + "/app")
    page.fill("#turns", "Fix the off-by-one error in pagination")
    page.fill("#diff", "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,3 @@\n def page(n):\n-    return n\n+    return n - 1\n")
    page.get_by_role("button", name="Analyse").click()
    page.get_by_test_id("summary").wait_for(timeout=180000)
    page.locator("article[aria-label^='Hunk ']").first.wait_for(timeout=30000)
    assert page.locator("article[aria-label^='Hunk ']").count() == 1


def test_analyse_disabled_without_diff(page, servers):
    page.goto(servers + "/app")
    assert page.get_by_role("button", name="Analyse").is_disabled()


@pytest.mark.parametrize("path,needle", [("/dev", "Developer console"), ("/dev/results", "Results"), ("/dev/experiments", "Experiments"), ("/dev/dataset", "Dataset"),
                                          ("/dev/testing", "Testing"), ("/dev/related-work", "Related work"), ("/about", "Sift"), ("/how-it-works", "How Sift works")])
def test_pages_load(page, servers, path, needle):
    resp = page.goto(servers + path)
    assert resp.status == 200
    page.wait_for_load_state("networkidle")
    assert page.locator("main, #root").first.inner_text().strip() != ""
    assert needle.lower() in page.locator("body").inner_text().lower()
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / f"e2e{path.replace('/', '_') or '_review'}_desktop.png"), full_page=True)
    assert not page.errors


def test_unknown_route_and_mobile(browser_ctx, servers):
    ctx = browser_ctx.new_context(viewport={"width": 390, "height": 800})
    pg = ctx.new_page()
    pg.goto(servers + "/nope")
    assert "not found" in pg.locator("body").inner_text().lower()
    pg.goto(servers + "/app")
    pg.wait_for_load_state("networkidle")
    assert pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    SHOTS.mkdir(parents=True, exist_ok=True)
    pg.screenshot(path=str(SHOTS / "e2e_review_mobile.png"))
    ctx.close()


def test_inputs_panel_toggle_and_persistence(page, servers):
    page.goto(servers + "/app")
    btn = page.get_by_test_id("inputs-toggle")
    panel = page.get_by_test_id("inputs-panel")
    assert btn.get_attribute("aria-expanded") == "true"
    assert panel.is_visible()
    w_open = page.locator("section[aria-label='Ranked hunks']").bounding_box()["width"]
    btn.click()
    assert btn.get_attribute("aria-expanded") == "false"
    page.wait_for_function("document.querySelector('[data-testid=inputs-panel]').getBoundingClientRect().height < 2")
    page.wait_for_timeout(350)
    assert not panel.is_visible()
    assert page.locator("section[aria-label='Ranked hunks']").bounding_box()["width"] > w_open + 150
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    assert page.evaluate("localStorage.getItem('sift-inputs-open')") == "0"
    page.reload()
    page.get_by_test_id("inputs-toggle").wait_for()
    assert page.get_by_test_id("inputs-toggle").get_attribute("aria-expanded") == "false"
    assert not page.get_by_test_id("inputs-panel").is_visible()
    # keyboard shortcut reopens it, and j/k/a/f/d are unaffected
    page.locator("body").click(position={"x": 700, "y": 600})
    page.keyboard.press("[")
    assert page.get_by_test_id("inputs-toggle").get_attribute("aria-expanded") == "true"
    page.wait_for_timeout(350)
    assert page.get_by_test_id("inputs-panel").is_visible()
    assert page.evaluate("localStorage.getItem('sift-inputs-open')") == "1"
    assert not page.errors


def test_inputs_panel_mobile_collapses_after_analyse(browser_ctx, servers):
    ctx = browser_ctx.new_context(viewport={"width": 390, "height": 800})
    pg = ctx.new_page()
    pg.goto(servers + "/app")
    btn = pg.get_by_test_id("inputs-toggle")
    assert btn.get_attribute("aria-expanded") == "true"
    pg.fill("#turns", "Fix the off-by-one error in pagination")
    pg.fill("#diff", "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,3 @@\n def page(n):\n-    return n\n+    return n - 1\n")
    pg.get_by_role("button", name="Analyse").click()
    pg.get_by_test_id("summary").wait_for(timeout=180000)
    pg.wait_for_function("document.querySelector('[data-testid=inputs-toggle]').getAttribute('aria-expanded') === 'false'")
    btn.click()
    assert btn.get_attribute("aria-expanded") == "true"
    assert pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    ctx.close()


def test_related_work_page_baselines_from_results(page, servers):
    page.goto(servers + "/dev/related-work")
    page.get_by_role("heading", name="Related work").wait_for()
    assert "not comparable" in page.get_by_test_id("caveat").inner_text()
    tbl = page.get_by_role("table", name="Baselines on Sift-IA, test split")
    tbl.wait_for(timeout=15000)
    rows = tbl.locator("tbody tr")
    assert rows.count() == 7
    main = tbl.locator("tbody tr", has_text="ModernBERT cross-encoder (main)")
    cells = main.locator("td").all_inner_texts()
    assert all(c.replace(".", "").isdigit() for c in cells[1:]), cells
    # numbers come from the results file, not from source
    import json
    m = json.loads((ROOT / "results" / "master_comparison.json").read_text())
    row = next(r for r in m["rows"] if r["model"].startswith("Main"))
    assert cells[3] == f"{row['auc_hard']:.3f}"
    assert page.get_by_test_id("refs-confirmed").locator("a").count() == 6
    assert page.get_by_test_id("refs-memory").locator("li").count() == 6
    assert "not re-verified" in page.get_by_test_id("refs-memory").inner_text()
    assert page.get_by_role("link", name="Related work").is_visible()
    assert not page.errors


@pytest.mark.parametrize("path", ["/dev", "/dev/results", "/dev/experiments", "/dev/dataset", "/dev/testing", "/dev/related-work"])
def test_every_page_shows_highlights(page, servers, path):
    page.goto(servers + path)
    strip = page.get_by_test_id("highlights")
    strip.wait_for(timeout=30000)
    txt = strip.inner_text().lower()
    assert "what worked best" in txt
    assert "criterion" in txt
    assert page.get_by_test_id("highlights-body").is_visible()
    assert not page.errors


def test_product_pages_have_no_developer_strip(page, servers):
    for path in ["/", "/app", "/about", "/how-it-works"]:
        page.goto(servers + path)
        page.wait_for_load_state("networkidle")
        assert page.get_by_test_id("highlights").count() == 0, path


def test_highlights_collapse_state_persists(page, servers):
    page.goto(servers + "/dev/results")
    page.get_by_test_id("highlights-body").wait_for()
    page.get_by_test_id("highlights-toggle").click()
    assert page.get_by_test_id("highlights-body").count() == 0
    assert page.evaluate("localStorage.getItem('sift-highlights-collapsed')") == "1"
    page.goto(servers + "/dev/dataset")  # shared across the non-Review pages
    page.get_by_test_id("highlights").wait_for()
    assert page.get_by_test_id("highlights-body").count() == 0
    page.reload()
    page.get_by_test_id("highlights").wait_for()
    assert page.get_by_test_id("highlights-body").count() == 0
    page.get_by_test_id("highlights-toggle").click()
    assert page.get_by_test_id("highlights-body").is_visible()


def test_highlights_values_come_from_results_files(page, servers):
    import json
    m = json.loads((ROOT / "results" / "master_comparison.json").read_text())
    main = next(r for r in m["rows"] if r["group"] == "main")
    page.goto(servers + "/dev")
    page.get_by_test_id("highlights-body").wait_for()
    assert f"{main['auc_hard']:.3f}" in page.get_by_test_id("highlights-body").inner_text()


def test_landing_renders_and_proof_matches_results(page, servers):
    import json
    m = json.loads((ROOT / "results" / "master_comparison.json").read_text())["rows"]
    main = next(r for r in m if r["group"] == "main")
    base = max((r for r in m if r["group"] == "non_neural" and r.get("auc_hard") is not None), key=lambda r: r["auc_hard"])
    o = json.loads((ROOT / "results" / "latency_size_onnx.json").read_text())["student_onnx_int8"]
    page.goto(servers + "/")
    page.get_by_role("heading", level=1).wait_for()
    assert "hunks" in page.get_by_role("heading", level=1).inner_text().lower()
    page.get_by_test_id("proof").wait_for(timeout=15000)
    assert page.get_by_test_id("proof-auc").inner_text() == f"{main['auc_hard']:.3f}"
    assert page.get_by_test_id("proof-baseline").inner_text() == f"{base['auc_hard']:.3f}"
    assert page.get_by_test_id("proof-size").inner_text() == f"{o['size_mb']:.1f}"
    assert page.get_by_test_id("proof-p95").inner_text() == f"{o['latency_ms']['p95_ms']:.0f}"
    assert "single seed" in page.locator("body").inner_text()
    assert page.get_by_role("link", name="Developers").first.get_attribute("href") == "/dev"
    assert not page.errors


def test_cta_reaches_workspace(page, servers):
    page.goto(servers + "/")
    page.get_by_role("link", name="Try Sift").first.click()
    page.wait_for_url("**/app")
    page.get_by_role("button", name="Analyse").wait_for()


@pytest.mark.parametrize("old,new", [("/results", "/dev/results"), ("/experiments", "/dev/experiments"), ("/dataset", "/dev/dataset"),
                                      ("/testing", "/dev/testing"), ("/related-work", "/dev/related-work"), ("/review", "/app")])
def test_old_routes_redirect(page, servers, old, new):
    page.goto(servers + old)
    page.wait_for_url("**" + new)


def test_dev_console_shell_and_back_link(page, servers):
    page.goto(servers + "/dev/testing")
    assert page.get_by_text("Developer console").first.is_visible()
    page.get_by_role("link", name="Back to product").click()
    page.wait_for_url(servers + "/")


def test_testing_raw_tree_behind_toggle(page, servers):
    page.goto(servers + "/dev/testing")
    page.get_by_test_id("raw-toggle").wait_for(timeout=15000)
    assert page.get_by_test_id("raw-tree").count() == 0
    page.get_by_test_id("raw-toggle").click()
    assert page.get_by_test_id("raw-tree").is_visible()


def test_theme_toggle_in_both_shells(page, servers):
    for path in ["/", "/dev"]:
        page.goto(servers + path)
        before = page.evaluate("document.documentElement.classList.contains('dark')")
        page.locator("button[title='Toggle theme']").first.click()
        assert page.evaluate("document.documentElement.classList.contains('dark')") != before


def test_mobile_product_menu(browser_ctx, servers):
    ctx = browser_ctx.new_context(viewport={"width": 390, "height": 800})
    pg = ctx.new_page()
    pg.goto(servers + "/")
    pg.get_by_role("button", name="Menu").click()
    assert pg.get_by_role("navigation", name="Mobile").is_visible()
    assert pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    ctx.close()
