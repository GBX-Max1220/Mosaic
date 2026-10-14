import { test, expect } from '@playwright/test';

async function startLoop(page, width = 1440) {
  await page.setViewportSize({ width, height: 1000 });
  const now = new Date('2026-10-13T00:00:00Z');
  await page.clock.install({ time: now });
  await page.clock.pauseAt(now);
  await page.goto('/');
  const loop = page.locator('[data-definition-loop]');
  await expect(loop).toHaveCount(1, { timeout: 1000 });
  await loop.scrollIntoViewIfNeeded();
  await expect(loop).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(1200);
  return loop;
}

const visibleUnits = loop => loop.locator('.definition-live .definition-unit').evaluateAll(nodes => nodes
  .filter(node => Number(getComputedStyle(node).opacity) > .9)
  .map(node => ({ index: node.dataset.sessionIndex, kind: node.dataset.kind })));

test('a rollout grows sessions before one-to-one segment conversion and a shared policy update', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const loop = await startLoop(page);
  const first = await loop.locator('.definition-live .definition-unit').first().elementHandle();
  expect(await visibleUnits(loop)).toEqual([{ index: '1', kind: 'session' }]);
  await page.clock.fastForward(3000);
  expect(await visibleUnits(loop)).toEqual([{ index: '1', kind: 'session' }, { index: '2', kind: 'session' }]);
  await page.clock.fastForward(4600); // 8800: complete, still sessions.
  await expect(loop).toHaveAttribute('data-phase', 'completed');
  expect(await visibleUnits(loop)).toEqual(['1', '2', 'n'].map(index => ({ index, kind: 'session' })));
  expect(await loop.locator('.definition-live .definition-trainer').evaluate(node => Number(node.style.opacity))).toBe(0);
  await page.clock.fastForward(4000); // 12800: same three units become segments.
  await expect(loop).toHaveAttribute('data-phase', 'segments');
  expect(await visibleUnits(loop)).toEqual(['1', '2', 'n'].map(index => ({ index, kind: 'segment' })));
  expect(await loop.locator('.definition-live .definition-unit').first().evaluate((node, original) => node === original, first)).toBe(true);
  const advantages = loop.locator('.definition-live .definition-advantage');
  await expect(advantages).toHaveText(['A1', 'A2', 'An']);
  expect(await advantages.evaluateAll(nodes => nodes.every(node => Number(node.style.opacity) > .9))).toBe(true);
  await page.clock.fastForward(4500); // 17300: all three feed one trainer.
  await expect(loop).toHaveAttribute('data-phase', 'training');
  await expect(loop.locator('.definition-live .definition-trainer')).toHaveAttribute('data-updated', 'true');
  const contributions = loop.locator('.definition-live [data-contribution]');
  await expect(contributions).toHaveCount(3);
  expect(await contributions.evaluateAll(nodes => nodes.every(node => node.dataset.delivered === 'true'))).toBe(true);
  expect(errors).toEqual([]);
});

test('definition captions fade and a new loop clears all prior segments and advantages', async ({ page }) => {
  const loop = await startLoop(page);
  const message = loop.locator('.definition-loop-message');
  await page.clock.fastForward(6650); // 7850.
  expect(await message.evaluate(node => Number(node.style.opacity))).toBeLessThan(.8);
  await page.clock.fastForward(300);
  await expect(loop).toHaveAttribute('data-phase', 'completed');
  expect(await message.evaluate(node => Number(node.style.opacity))).toBeLessThan(.8);
  await page.clock.fastForward(14350); // 22500: second loop.
  await expect(loop).toHaveAttribute('data-cycle', '1');
  await expect(loop).toHaveAttribute('data-phase', 'rollout');
  expect(await visibleUnits(loop)).toEqual([{ index: '1', kind: 'session' }]);
  expect(await loop.locator('.definition-live .definition-advantage').evaluateAll(nodes => nodes.every(node => Number(node.style.opacity) === 0))).toBe(true);
  expect(await loop.locator('.definition-live .definition-trainer').evaluate(node => Number(node.style.opacity))).toBe(0);
});

test('definition playback freezes offscreen and in a background tab', async ({ page }) => {
  const loop = await startLoop(page);
  await page.clock.fastForward(3000);
  await page.locator('#takeaways').scrollIntoViewIfNeeded();
  await expect(loop).toHaveAttribute('data-state', 'paused');
  const paused = await loop.innerHTML();
  await page.clock.fastForward(10000);
  expect(await loop.innerHTML()).toBe(paused);
  await loop.scrollIntoViewIfNeeded();
  await expect(loop).toHaveAttribute('data-state', 'playing');
  await page.evaluate(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, value: true });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(loop).toHaveAttribute('data-state', 'paused');
  const background = await loop.innerHTML();
  await page.clock.fastForward(10000);
  expect(await loop.innerHTML()).toBe(background);
  await page.evaluate(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, value: false });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(loop).toHaveAttribute('data-state', 'playing');
  await page.clock.fastForward(5000);
  await expect(loop).toHaveAttribute('data-phase', 'completed');
});

for (const width of [360, 390, 768, 1440]) {
  test(`the embedded definition loop stays within its card at ${width}px`, async ({ page }) => {
    const loop = await startLoop(page, width);
    let elapsed = 1200;
    for (const moment of [1200, 4200, 6900, 8800, 12300, 14500, 15800, 17300, 19500]) {
      await page.clock.fastForward(moment - elapsed);
      const geometry = await loop.evaluate(node => {
        const scene = node.querySelector('.definition-live').getBoundingClientRect();
        const units = [...node.querySelectorAll('.definition-live .definition-unit')].filter(unit => Number(unit.style.opacity) > .1);
        const visible = units.flatMap(unit => [unit, ...[...unit.children].filter(child => Number(getComputedStyle(child).opacity) > .1)]);
        for (const extra of node.querySelectorAll('.definition-live .definition-trainer, .definition-live .definition-packet')) {
          if (Number(extra.style.opacity) > .1) visible.push(extra);
        }
        const rollout = node.querySelector('.definition-rollout');
        const rolloutRect = rollout.getBoundingClientRect();
        const uncentered = Number(rollout.style.opacity) > .9 ? units.filter(unit => {
          if (unit.dataset.kind !== 'session') return false;
          const rect = unit.getBoundingClientRect();
          return Math.abs(rect.top + rect.height / 2 - rolloutRect.top - rolloutRect.height / 2) > 1;
        }).map(unit => unit.dataset.sessionIndex) : [];
        let trainingAlignment = 0;
        const trainer = node.querySelector('.definition-live .definition-trainer');
        if (Number(trainer.style.opacity) > .9) {
          const centers = units.map(unit => {
            const rect = unit.getBoundingClientRect();
            return rect.left + rect.width / 2;
          });
          const rect = trainer.getBoundingClientRect();
          const center = rect.left + rect.width / 2;
          const path = node.querySelectorAll('[data-contribution]')[1];
          trainingAlignment = Math.max(Math.abs(centers[1] - center),
            Math.abs(centers[0] + centers[2] - 2 * center),
            Math.abs(path.getPointAtLength(0).x - path.getPointAtLength(path.getTotalLength()).x));
        }
        return { bad: visible.filter(item => {
          const rect = item.getBoundingClientRect();
          return rect.left < scene.left - 1 || rect.right > scene.right + 1 || rect.top < scene.top - 1 || rect.bottom > scene.bottom + 1;
        }).map(item => item.className), uncentered, trainingAlignment, overflow: document.documentElement.scrollWidth > innerWidth,
        oldEquation: document.querySelectorAll('.setting-definition [role="math"]').length };
      });
      expect(geometry.bad, `geometry at ${moment}ms`).toEqual([]);
      expect(geometry.uncentered, `session centering at ${moment}ms`).toEqual([]);
      expect(geometry.trainingAlignment, `trainer symmetry at ${moment}ms`).toBeLessThan(1);
      expect(geometry.overflow).toBe(false);
      expect(geometry.oldEquation).toBe(0);
      elapsed = moment;
    }
    await page.setViewportSize({ width: width === 360 ? 1440 : 360, height: 1000 });
    await loop.scrollIntoViewIfNeeded();
    await page.clock.runFor(32);
    await expect(loop).toHaveAttribute('data-cycle', '0');
    await expect(loop).toHaveAttribute('data-phase', 'reset');
  });
}

test('reduced motion shows a complete static mapping and switching back begins a fresh rollout', async ({ page }) => {
  const loop = await startLoop(page);
  await page.clock.fastForward(6650);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(loop).toHaveAttribute('data-state', 'static');
  await expect(loop.locator('.definition-static')).toBeVisible();
  await expect(loop.locator('.definition-live')).toBeHidden();
  await expect(loop.locator('.definition-static .definition-static-segment')).toHaveCount(3);
  expect(await loop.locator('.definition-loop-message').evaluate(node => Number(getComputedStyle(node).opacity))).toBe(1);
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await loop.scrollIntoViewIfNeeded();
  await expect(loop).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(1200);
  await expect(loop).toHaveAttribute('data-cycle', '0');
  expect(await visibleUnits(loop)).toEqual([{ index: '1', kind: 'session' }]);
});

test('the definition remains complete without JavaScript', async ({ browser }) => {
  const page = await browser.newPage({ javaScriptEnabled: false, viewport: { width: 360, height: 1000 } });
  try {
    await page.goto('http://127.0.0.1:4173/');
    const loop = page.locator('[data-definition-loop]');
    await expect(loop).toHaveCount(1, { timeout: 1000 });
    await expect(loop).toHaveAccessibleName(/rollout.*sessions.*training segment.*policy update/i);
    await expect(loop.locator('.definition-static .definition-static-session')).toHaveCount(3);
    await expect(loop.locator('.definition-static .definition-static-segment')).toHaveCount(3);
    await expect(loop.locator('.definition-static .definition-trainer')).toBeVisible();
    expect(await loop.evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true);
  } finally { await page.close(); }
});

test('a browser without IntersectionObserver keeps the static definition mapping', async ({ page }) => {
  await page.addInitScript(() => { delete window.IntersectionObserver; });
  await page.goto('/');
  const loop = page.locator('[data-definition-loop]');
  await expect(loop).toHaveCount(1, { timeout: 1000 });
  await expect(loop).toHaveAttribute('data-state', 'static');
  await expect(loop.locator('.definition-static .definition-static-segment')).toHaveCount(3);
});

test('a returning reader with an old unversioned stylesheet sees positioned animation cards', async ({ page }) => {
  const response = await page.request.get('/static/css/style.css');
  const staleStyles = (await response.text()).split('\n')
    .filter(line => !line.trim().startsWith('.definition-')).join('\n');
  // Reproduce a cached stylesheet from before the definition animation existed.
  await page.route(url => url.pathname === '/static/css/style.css' && !url.search,
    route => route.fulfill({ contentType: 'text/css', body: staleStyles }));
  const loop = await startLoop(page);
  await expect(loop.locator('.definition-static')).toBeHidden();
  expect(await loop.locator('.definition-live').evaluate(node => getComputedStyle(node).position)).toBe('relative');
  expect(await loop.locator('.definition-unit').first().evaluate(node => getComputedStyle(node).position)).toBe('absolute');
});
