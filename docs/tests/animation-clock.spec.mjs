import { test, expect } from '@playwright/test';

const loops = [
  { name: 'single-session', selector: '[data-session-loop]', boundary: 3000, phase: 'interaction' },
  { name: 'compaction', selector: '#compaction-animation', boundary: 7000, phase: 'boundary' },
  { name: 'forking', selector: '#forking-animation', boundary: 6000, phase: 'fork' },
  { name: 'definition', selector: '[data-definition-loop]', boundary: 8000, phase: 'completed' },
  { name: 'trajectory advantage', selector: '[data-advantage-loop]', boundary: 6500, phase: 'waiting' },
];

for (const width of [390, 1440]) {
  for (const { name, selector, boundary, phase } of loops) {
    test(`${name} starts and resumes when frame timestamps precede the request at ${width}px`, async ({ page }) => {
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.setViewportSize({ width, height: 900 });
      const now = new Date('2026-10-15T00:00:00Z');
      await page.clock.install({ time: now });
      await page.clock.pauseAt(now);
      await page.addInitScript(() => {
        // Reproduce the native Chromium timing observed under CPU contention:
        // a frame's timestamp can precede performance.now() at its request.
        const requestFrame = window.requestAnimationFrame.bind(window);
        window.requestAnimationFrame = callback => requestFrame(timestamp => callback(timestamp - 24));
      });
      await page.goto('/');
      await page.clock.fastForward(100);
      const loop = page.locator(selector);
      await loop.scrollIntoViewIfNeeded();
      await expect(loop).toHaveAttribute('data-state', 'playing');
      await page.clock.runFor(64);
      expect(errors, 'the first frame must not crash the stage lookup').toEqual([]);
      await page.clock.fastForward(boundary + 100);
      await expect(loop).toHaveAttribute('data-phase', phase);

      await page.evaluate(() => {
        Object.defineProperty(document, 'hidden', { configurable: true, value: true });
        document.dispatchEvent(new Event('visibilitychange'));
      });
      await expect(loop).toHaveAttribute('data-state', 'paused');
      const paused = await loop.innerHTML();
      await page.clock.fastForward(20000);
      expect(await loop.innerHTML(), 'background time must not advance the animation').toBe(paused);
      await page.evaluate(() => {
        Object.defineProperty(document, 'hidden', { configurable: true, value: false });
        document.dispatchEvent(new Event('visibilitychange'));
      });
      await expect(loop).toHaveAttribute('data-state', 'playing');
      await page.clock.runFor(64);
      await expect(loop).toHaveAttribute('data-phase', phase);
      await page.clock.fastForward(1000);
      expect(await loop.innerHTML(), 'animation must keep advancing after resume').not.toBe(paused);
      expect(errors).toEqual([]);
    });
  }
}
