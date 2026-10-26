import sys
import os
import asyncio
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding="utf-8")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 1000})
        await page.goto("http://localhost:8501", timeout=30000)
        await page.wait_for_selector(".hero-title", timeout=15000)

        buttons = page.locator("button")
        count = await buttons.count()
        print(f"Total buttons: {count}")
        for i in range(count):
            txt = await buttons.nth(i).inner_text()
            if "Sample" in txt or "#1" in txt:
                print(f"Clicking button {i}: '{txt}'")
                await buttons.nth(i).click()
                await page.wait_for_timeout(6000)
                await page.screenshot(path="outputs/app_screenshot_sample1.png", full_page=True)
                print("Saved outputs/app_screenshot_sample1.png")
                break

        # Check tabs
        tabs = page.locator("[data-baseweb='tab']")
        tab_count = await tabs.count()
        print(f"Total tabs found: {tab_count}")
        for i in range(tab_count):
            t_txt = await tabs.nth(i).inner_text()
            print(f"Tab {i}: '{t_txt}'")
            if "Pipeline" in t_txt:
                print(f"Clicking Tab {i}...")
                await tabs.nth(i).click()
                await page.wait_for_timeout(2000)
                await page.screenshot(path="outputs/app_screenshot_pipeline_tab.png", full_page=True)
                print("Saved outputs/app_screenshot_pipeline_tab.png")
            elif "Symbol" in t_txt:
                print(f"Clicking Tab {i}...")
                await tabs.nth(i).click()
                await page.wait_for_timeout(2000)
                await page.screenshot(path="outputs/app_screenshot_symbols_tab.png", full_page=True)
                print("Saved outputs/app_screenshot_symbols_tab.png")
            elif "Insights" in t_txt:
                print(f"Clicking Tab {i}...")
                await tabs.nth(i).click()
                await page.wait_for_timeout(2000)
                await page.screenshot(path="outputs/app_screenshot_insights_tab.png", full_page=True)
                print("Saved outputs/app_screenshot_insights_tab.png")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
