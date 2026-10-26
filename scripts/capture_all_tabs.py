import sys
import asyncio
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding="utf-8")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 1100})
        await page.goto("http://localhost:8501", timeout=30000)
        await page.wait_for_selector(".hero-title", timeout=15000)

        # Click Sample #1 button
        sample1_btn = page.locator("button:has-text('Sample #1')")
        print("Clicking Sample #1 button...")
        await sample1_btn.click()

        # Wait for results metric row to appear
        print("Waiting for .metric-row...")
        await page.wait_for_selector(".metric-row", timeout=15000)
        await page.wait_for_timeout(2000)

        # Capture Tab 1 (Solution)
        await page.screenshot(path="outputs/screenshot_tab1_solution.png", full_page=True)
        print("Captured outputs/screenshot_tab1_solution.png")

        # Find all tab buttons
        tab_buttons = page.locator("[role='tab']")
        count = await tab_buttons.count()
        print(f"Result tabs found: {count}")

        for i in range(count):
            txt = await tab_buttons.nth(i).inner_text()
            print(f"Tab {i}: '{txt}'")
            if "CV Pipeline" in txt or "Pipeline" in txt:
                print(f"Clicking Tab {i} (Pipeline)...")
                await tab_buttons.nth(i).click()
                await page.wait_for_timeout(2500)
                await page.screenshot(path="outputs/screenshot_tab2_pipeline.png", full_page=True)
                print("Captured outputs/screenshot_tab2_pipeline.png")
            elif "Symbol" in txt:
                print(f"Clicking Tab {i} (Symbols)...")
                await tab_buttons.nth(i).click()
                await page.wait_for_timeout(2500)
                await page.screenshot(path="outputs/screenshot_tab3_symbols.png", full_page=True)
                print("Captured outputs/screenshot_tab3_symbols.png")
            elif "Model" in txt or "Insights" in txt:
                print(f"Clicking Tab {i} (Insights)...")
                await tab_buttons.nth(i).click()
                await page.wait_for_timeout(2500)
                await page.screenshot(path="outputs/screenshot_tab4_insights.png", full_page=True)
                print("Captured outputs/screenshot_tab4_insights.png")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
