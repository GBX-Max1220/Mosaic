import { test, expect } from '@playwright/test';

async function startLoop(page, width = 1440) {
  await page.setViewportSize({ width, height: 1000 });
  const now = new Date('2026-10-13T00:00:00Z');
  await page.clock.install({ time: now });
  await page.clock.pauseAt(now);
  await page.goto('/');
  const loop = page.locator('[data-advantage-loop]');
  await expect(loop).toHaveCount(1, { timeout: 1000 });
  await loop.scrollIntoViewIfNeeded();
  await expect(loop).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(800);
  return loop;
}

const readGroup = (loop, outcome) => loop.locator(`.advantage-live-card[data-outcome="${outcome}"]`).evaluateAll(nodes => nodes
  .sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top)
  .map(node => ({ id: node.dataset.rolloutId, count: Number(node.dataset.sessions),
    rank: Number(node.dataset.rank), weight: Number(node.dataset.weight), advantage: Number(node.dataset.advantage) })));

for (const width of [360, 768, 1440]) {
  test(`sampling uses balanced centered rollout cards and elongated sessions before shrinking into subgroups at ${width}px`, async ({ page }) => {
    const loop = await startLoop(page, width);
    await page.clock.fastForward(9900); // 10700: all session cards have fully appeared.
    const geometry = () => loop.evaluate(node => {
      const scene = node.querySelector('.advantage-live').getBoundingClientRect();
      const card = node.querySelector('.advantage-live-card[data-rollout-id="5"]').getBoundingClientRect();
      const blocks = [...node.querySelectorAll('.advantage-live-card[data-rollout-id="5"] .advantage-session-block')]
        .map(block => block.getBoundingClientRect());
      return { sceneWidth: scene.width, sceneCenter: scene.left + scene.width / 2,
        cardWidth: card.width, cardCenter: card.left + card.width / 2,
        rowWidth: blocks.at(-1).right - blocks[0].left,
        blockWidth: blocks[0].width, blockHeight: blocks[0].height };
    });
    const sampled = await geometry();
    expect(sampled.cardCenter).toBeCloseTo(sampled.sceneCenter, 0);
    if (width >= 768) expect(sampled.cardWidth).toBeLessThan(sampled.sceneWidth * .8);
    expect(sampled.blockHeight).toBeGreaterThanOrEqual(11);
    expect(sampled.blockWidth / sampled.blockHeight).toBeGreaterThan(1.8);
    expect(sampled.rowWidth / sampled.cardWidth, 'eight sessions should occupy a meaningful part of the rollout').toBeGreaterThan(.55);
    expect(sampled.rowWidth / sampled.cardWidth).toBeLessThan(.9);
    await page.clock.fastForward(5900); // 16600: session widths interpolate during the visible slide.
    const moving = await geometry();
    expect(moving.blockWidth).toBeLessThan(sampled.blockWidth);
    expect(moving.blockWidth).toBeGreaterThan(10);
    await page.clock.fastForward(7100); // 23700: retain the accepted compact two-column layout.
    const grouped = await geometry();
    expect(grouped.blockWidth).toBeLessThanOrEqual(10);
    expect(grouped.blockHeight).toBeCloseTo(grouped.blockWidth, 1);
    expect((await readGroup(loop, 1)).map(row => row.advantage)).toEqual([1.2, 1, .8]);
  });
}

test('six rollouts appear together and session counts wait for the last session of the longest rollout', async ({ page }) => {
  const loop = await startLoop(page);
  const cards = loop.locator('.advantage-live-card');
  const counts = () => cards.locator('.advantage-session-count').evaluateAll(nodes => nodes.map(node => Number(getComputedStyle(node).opacity)));
  const sessions = () => cards.evaluateAll(nodes => nodes.map(node => ({
    id: node.dataset.rolloutId, complete: node.dataset.complete,
    visible: [...node.querySelectorAll('.advantage-session-block')].filter(block => Number(getComputedStyle(block).opacity) > .9).length,
  })));
  await expect(cards).toHaveCount(6);
  expect(await cards.evaluateAll(nodes => nodes.map(node => Number(getComputedStyle(node).opacity)))).toEqual([1, 1, 1, 1, 1, 1]);
  expect(await counts()).toEqual([0, 0, 0, 0, 0, 0]);
  expect((await sessions()).map(row => row.visible)).toEqual([0, 0, 0, 0, 0, 0]);
  await page.clock.fastForward(2500); // 3300: short rollouts are complete while long ones still grow.
  const generating = await sessions();
  expect(generating.find(row => row.id === '2')).toMatchObject({ complete: 'true', visible: 2 });
  expect(generating.find(row => row.id === '4').complete).toBe('false');
  await page.clock.fastForward(5000); // 8300: five rollouts wait for Rollout 4.
  await expect(loop).toHaveAttribute('data-phase', 'waiting');
  const waiting = await sessions();
  expect(waiting.filter(row => row.complete === 'true')).toHaveLength(5);
  expect(waiting.find(row => row.id === '4').visible).toBeLessThan(8);
  expect(await counts()).toEqual([0, 0, 0, 0, 0, 0]);
  await page.clock.fastForward(1200); // 9500: even during the final session's entrance, no N is shown.
  expect((await sessions()).find(row => row.id === '4').complete).toBe('false');
  expect(await counts()).toEqual([0, 0, 0, 0, 0, 0]);
  await page.clock.fastForward(1200); // 10700: the whole group is complete, and all counts appear together.
  expect((await sessions()).map(row => row.visible)).toEqual([4, 2, 2, 8, 8, 4]);
  expect(await counts()).toEqual([1, 1, 1, 1, 1, 1]);
  await expect(cards.locator('.advantage-session-count')).toHaveText(['N = 4', 'N = 2', 'N = 2', 'N = 8', 'N = 8', 'N = 4']);
});

test('scoring keeps cards in place before a visible smooth slide into sorted outcome subgroups', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const loop = await startLoop(page);
  const original = await loop.locator('.advantage-live-card').first().elementHandle();
  const geometry = () => loop.locator('.advantage-live-card').evaluateAll(nodes => nodes.map(node => {
    const rect = node.getBoundingClientRect();
    return { x: rect.left, y: rect.top, width: rect.width, opacity: Number(getComputedStyle(node).opacity),
      color: getComputedStyle(node).backgroundColor };
  }));
  await page.clock.fastForward(9900); // 10700: sampling has finished, positions remain fixed for scoring.
  const unscored = await geometry();
  await page.clock.fastForward(3800); // 14500: green/red and R appear before any movement.
  await expect(loop).toHaveAttribute('data-phase', 'scoring');
  const scored = await geometry();
  scored.forEach((card, index) => {
    expect(card.x).toBe(unscored[index].x);
    expect(card.y).toBe(unscored[index].y);
    expect(card.color).not.toBe(unscored[index].color);
  });
  for (const outcome of [1, 0]) {
    const rewards = loop.locator(`.advantage-live-card[data-outcome="${outcome}"] .advantage-reward`);
    await expect(rewards).toHaveText([`R = ${outcome}`, `R = ${outcome}`, `R = ${outcome}`]);
    expect(await rewards.evaluateAll(nodes => nodes.every(node => Number(getComputedStyle(node).opacity) === 1))).toBe(true);
  }
  await page.clock.fastForward(2100); // 16600: all six cards remain visible while sliding, rather than swapping under a fade.
  const moving = await geometry();
  expect(moving.every(card => card.opacity === 1)).toBe(true);
  expect(moving.some((card, index) => Math.hypot(card.x - scored[index].x, card.y - scored[index].y) > 5)).toBe(true);
  await page.clock.fastForward(300); // a later frame must continue moving smoothly.
  const later = await geometry();
  expect(later.some((card, index) => Math.hypot(card.x - moving[index].x, card.y - moving[index].y) > 5)).toBe(true);
  await page.clock.fastForward(1900); // 18800: both subgroups are sorted, and R has disappeared.
  for (const outcome of [1, 0]) {
    const rows = await readGroup(loop, outcome);
    expect(rows.map(row => row.count)).toEqual([2, 4, 8]);
    expect(rows.map(row => row.rank)).toEqual([-1, 0, 1]);
  }
  const columns = await loop.locator('.advantage-live-card').evaluateAll(nodes => {
    const success = nodes.filter(node => node.dataset.outcome === '1').map(node => node.getBoundingClientRect());
    const failure = nodes.filter(node => node.dataset.outcome === '0').map(node => node.getBoundingClientRect());
    return { right: Math.max(...success.map(rect => rect.right)), left: Math.min(...failure.map(rect => rect.left)) };
  });
  expect(columns.right).toBeLessThan(columns.left);
  expect(await loop.locator('.advantage-reward').evaluateAll(nodes => nodes.every(node => Number(getComputedStyle(node).opacity) === 0))).toBe(true);
  await expect(loop.locator('.advantage-rank, .advantage-weight')).toHaveCount(0);
  expect(await loop.locator('.advantage-live-card').first().evaluate((node, first) => node === first, original)).toBe(true);
  expect(errors).toEqual([]);
});

test('baseline advantages and translucent preference arrows precede sign-preserving advantage changes', async ({ page }) => {
  const loop = await startLoop(page);
  await page.clock.fastForward(19000); // 19800: original outcome advantages are readable before arrows appear.
  for (const outcome of [1, 0]) {
    expect((await readGroup(loop, outcome)).map(row => row.advantage)).toEqual(outcome ? [1, 1, 1] : [-1, -1, -1]);
  }
  expect(await loop.locator('.advantage-value').evaluateAll(nodes => nodes.every(node => Number(getComputedStyle(node).opacity) === 1))).toBe(true);
  expect(await loop.locator('.advantage-direction').evaluateAll(nodes => nodes.every(node => Number(getComputedStyle(node).opacity) === 0))).toBe(true);
  await page.clock.fastForward(1500); // 21300: green points upward, red downward; values are still unchanged.
  await expect(loop.locator('.advantage-direction[data-outcome="1"]')).toHaveAttribute('data-direction', 'up');
  await expect(loop.locator('.advantage-direction[data-outcome="0"]')).toHaveAttribute('data-direction', 'down');
  const opacity = await loop.locator('.advantage-direction').evaluateAll(nodes => nodes.map(node => Number(getComputedStyle(node).opacity)));
  expect(opacity).toHaveLength(2);
  opacity.forEach(value => { expect(value).toBeGreaterThan(.1); expect(value).toBeLessThan(.5); });
  expect((await readGroup(loop, 1)).map(row => row.advantage)).toEqual([1, 1, 1]);
  await page.clock.fastForward(1200); // 22500: advantage numbers interpolate, rather than jumping immediately to the result.
  const changing = (await readGroup(loop, 1)).map(row => row.advantage);
  expect(changing[0]).toBeGreaterThan(1);
  expect(changing[0]).toBeLessThan(1.2);
  await page.clock.fastForward(1200); // 23700: final credits retain the original outcome signs.
  await expect(loop).toHaveAttribute('data-phase', 'modulation');
  for (const outcome of [1, 0]) {
    expect((await readGroup(loop, outcome)).map(row => row.weight)).toEqual([1.2, 1, .8]);
  }
  expect((await readGroup(loop, 1)).map(row => row.advantage)).toEqual([1.2, 1, .8]);
  expect((await readGroup(loop, 0)).map(row => row.advantage)).toEqual([-1.2, -1, -.8]);
});

for (const width of [360, 1440]) {
  test(`score metadata shares a baseline and grouped cards keep credit close to sessions at ${width}px`, async ({ page }) => {
    const loop = await startLoop(page, width);
    let previous = 800;
    for (const time of [14500, 16600, 16950, 17300, 17950, 18200]) {
      await page.clock.fastForward(time - previous);
      previous = time;
      const metadata = await loop.locator('.advantage-live-card').evaluateAll(cards => cards.map(card => {
        const rect = card.getBoundingClientRect();
        const count = card.querySelector('.advantage-session-count').getBoundingClientRect();
        const reward = card.querySelector('.advantage-reward').getBoundingClientRect();
        const title = card.querySelector('.advantage-rollout-title').getBoundingClientRect();
        return { baselineGap: Math.abs(count.bottom - reward.bottom), gap: reward.left - count.right,
          collision: Math.min(title.right, count.right) - Math.max(title.left, count.left) > 1 &&
            Math.min(title.bottom, count.bottom) - Math.max(title.top, count.top) > 1,
          outside: count.left < rect.left || reward.right > rect.right || reward.bottom > rect.bottom };
      }));
      metadata.forEach(row => {
        expect(row.baselineGap, `N and R at ${time}`).toBeLessThan(1);
        expect(row.gap).toBeGreaterThanOrEqual(2);
        expect(row.collision).toBe(false);
        expect(row.outside).toBe(false);
      });
    }
    await page.clock.fastForward(5500); // 23700: final credit fits on one line, immediately below the sessions.
    const compact = await loop.locator('.advantage-live-card').evaluateAll(cards => cards.map(card => {
      const rect = card.getBoundingClientRect();
      const value = card.querySelector('.advantage-value').getBoundingClientRect();
      const label = card.querySelector('.advantage-value small').getBoundingClientRect();
      const number = card.querySelector('.advantage-value strong').getBoundingClientRect();
      const blocks = [...card.querySelectorAll('.advantage-session-block')].map(block => block.getBoundingClientRect());
      return { height: rect.height, blank: value.top - Math.max(...blocks.map(block => block.bottom)),
        inline: label.right <= number.left - 2 && Math.min(label.bottom, number.bottom) > Math.max(label.top, number.top),
        outside: value.left < rect.left || value.right > rect.right || value.bottom > rect.bottom };
    }));
    compact.forEach(card => {
      expect(card.height).toBeLessThanOrEqual(74);
      expect(card.blank).toBeGreaterThanOrEqual(3);
      expect(card.blank).toBeLessThanOrEqual(14);
      expect(card.inline).toBe(true);
      expect(card.outside).toBe(false);
    });
    const arrows = await loop.evaluate(root => [...root.querySelectorAll('.advantage-direction')].map(arrow => {
      const rect = arrow.getBoundingClientRect();
      const cards = [...root.querySelectorAll(`.advantage-live-card[data-outcome="${arrow.dataset.outcome}"]`)]
        .map(card => card.getBoundingClientRect());
      return { startGap: rect.top - Math.min(...cards.map(card => card.top)),
        endGap: rect.bottom - Math.max(...cards.map(card => card.bottom)) };
    }));
    arrows.forEach(arrow => {
      expect(Math.abs(arrow.startGap)).toBeLessThan(1);
      expect(Math.abs(arrow.endGap)).toBeLessThan(1);
    });
  });

  test(`regrouping keeps all six cards and their session/reward contents unobscured at ${width}px`, async ({ page }) => {
    const loop = await startLoop(page, width);
    let previous = 800;
    for (const time of [14500, 16100, 16400, 16700, 16950, 17050, 17300, 17600, 17950, 18800]) {
      await page.clock.fastForward(time - previous);
      previous = time;
      const collisions = await loop.locator('.advantage-live-card').evaluateAll(nodes => {
        const intersects = (a, b) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 &&
          Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1;
        const overlaps = [];
        nodes.forEach((node, index) => {
          const rect = node.getBoundingClientRect();
          nodes.slice(index + 1).forEach(other => {
            if (intersects(rect, other.getBoundingClientRect())) overlaps.push(`${node.dataset.rolloutId}/${other.dataset.rolloutId}`);
          });
          const reward = node.querySelector('.advantage-reward');
          if (Number(getComputedStyle(reward).opacity) <= .1) return;
          node.querySelectorAll('.advantage-session-block').forEach(block => {
            if (Number(getComputedStyle(block).opacity) > .9 && intersects(reward.getBoundingClientRect(), block.getBoundingClientRect())) {
              overlaps.push(`reward/session in ${node.dataset.rolloutId}`);
            }
          });
        });
        return overlaps;
      });
      expect(collisions, `visible movement at ${time}ms`).toEqual([]);
    }
  });
}

test('subtitles fade and a new prompt group clears session blocks, scores and final credit', async ({ page }) => {
  const loop = await startLoop(page);
  const message = loop.locator('.advantage-loop-message');
  await page.clock.fastForward(9050); // 9850: group generation is about to end.
  expect(await message.evaluate(node => Number(node.style.opacity))).toBeLessThan(.8);
  await page.clock.fastForward(300);
  expect(await message.evaluate(node => Number(node.style.opacity))).toBeLessThan(.8);
  await page.clock.fastForward(20650); // 30800.
  await expect(loop).toHaveAttribute('data-cycle', '1');
  await expect(loop).toHaveAttribute('data-phase', 'sample');
  for (const className of ['advantage-session-count', 'advantage-session-block', 'advantage-reward', 'advantage-value', 'advantage-direction']) {
    expect(await loop.locator(`.advantage-live .${className}`).evaluateAll(nodes => nodes.every(node => Number(getComputedStyle(node).opacity) === 0))).toBe(true);
  }
});

test('the trajectory animation pauses its whole state offscreen and in background tabs', async ({ page }) => {
  const loop = await startLoop(page);
  await page.clock.fastForward(15600); // 16400: freeze the visible regrouping slide itself.
  await page.locator('#takeaways').scrollIntoViewIfNeeded();
  await expect(loop).toHaveAttribute('data-state', 'paused');
  const offscreen = await loop.innerHTML();
  await page.clock.fastForward(10000);
  expect(await loop.innerHTML()).toBe(offscreen);
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
  await page.clock.fastForward(4000);
  await expect(loop).toHaveAttribute('data-phase', 'advantage');
});

for (const width of [360, 390, 768, 1440]) {
  test(`both method illustrations stay readable at ${width}px`, async ({ page }) => {
    const loop = await startLoop(page, width);
    let elapsed = 800;
    for (const moment of [800, 3300, 8300, 10700, 14500, 16500, 17200, 18800, 19800, 21300, 23700, 27500]) {
      await page.clock.fastForward(moment - elapsed);
      const bad = await loop.evaluate(root => {
        const scene = root.querySelector('.advantage-live').getBoundingClientRect();
        const visible = [...root.querySelectorAll('.advantage-live-card, .advantage-live-label, .advantage-direction')]
          .filter(node => Number(node.style.opacity) > .1);
        return visible.filter(node => {
          const rect = node.getBoundingClientRect();
          return rect.left < scene.left - 1 || rect.right > scene.right + 1 || rect.top < scene.top - 1 || rect.bottom > scene.bottom + 1 ||
            [...node.children].some(child => {
              if (Number(getComputedStyle(child).opacity) <= .1) return false;
              const text = child.getBoundingClientRect();
              return text.left < rect.left - 1 || text.right > rect.right + 1 || text.top < rect.top - 1 || text.bottom > rect.bottom + 1;
            });
        }).map(node => node.className);
      });
      expect(bad, `geometry at ${moment}`).toEqual([]);
      elapsed = moment;
    }
    const shaping = page.locator('[data-session-shaping]');
    await expect(shaping).toHaveCount(1);
    await expect(shaping).toHaveAccessibleName(/information gain.*session type/i);
    await expect(shaping.locator('.shaping-type-group')).toHaveCount(3);
    await expect(shaping.locator('.shaping-session')).toHaveCount(0);
    await expect(shaping.locator('.shaping-direction-note, .shaping-terminal-note')).toHaveCount(0);
    expect(await shaping.evaluate(root => ({ overflow: document.documentElement.scrollWidth > innerWidth,
      clipped: [...root.querySelectorAll('.shaping-type-group, .shaping-context')].filter(node => {
        const card = node.getBoundingClientRect();
        const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
        let text;
        while ((text = walker.nextNode())) {
          if (!text.textContent.trim()) continue;
          const range = document.createRange();
          range.selectNodeContents(text);
          if ([...range.getClientRects()].some(rect => rect.left < card.left - 1 || rect.right > card.right + 1 || rect.top < card.top - 1 || rect.bottom > card.bottom + 1)) return true;
        }
        return false;
      }).map(node => node.className) }))).toEqual({ overflow: false, clipped: [] });
    await page.setViewportSize({ width: width === 360 ? 1440 : 360, height: 1000 });
    await loop.scrollIntoViewIfNeeded();
    await page.clock.runFor(32);
    await expect(loop).toHaveAttribute('data-phase', 'reset');
    await expect(loop).toHaveAttribute('data-cycle', '0');
  });
}

for (const width of [360, 1440]) {
  test(`session shaping explains local gain before fusion prose and the final equation at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/');
    const diagram = page.locator('[data-session-shaping]');
    const equation = page.locator('#method .equation');
    await expect(equation).toHaveCount(1);
    const figureBounds = await diagram.boundingBox();
    const equationBounds = await equation.boundingBox();
    expect(equationBounds.y, 'the final equation follows the local-gain diagram').toBeGreaterThan(figureBounds.y + figureBounds.height);
    await expect(diagram.locator('.shaping-type-group')).toHaveCount(3);
    await expect(diagram.locator('.shaping-global-signal, .shaping-credit-branches, .shaping-session')).toHaveCount(0);
    await expect(diagram).toHaveAccessibleName(/information gain.*session type/i);
    const prose = diagram.locator('xpath=following-sibling::*[1]');
    expect(await prose.evaluate(node => node.tagName)).toBe('P');
    await expect(prose).toContainText(/^Finally,.*trajectory-level and session-level signals.*weighted sum.*final advantage for each session:/);
    const proseBounds = await prose.boundingBox();
    expect(proseBounds.y).toBeGreaterThan(figureBounds.y + figureBounds.height);
    expect(proseBounds.y + proseBounds.height).toBeLessThan(equationBounds.y);
    expect(await prose.evaluate(node => node.nextElementSibling.classList.contains('equation'))).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
}

test('reduced motion presents the complete pipeline and switching back samples a fresh group', async ({ page }) => {
  const loop = await startLoop(page);
  await page.clock.fastForward(20500);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(loop).toHaveAttribute('data-state', 'static');
  await expect(loop.locator('.advantage-static')).toBeVisible();
  await expect(loop.locator('.advantage-live')).toBeHidden();
  await expect(loop.locator('.advantage-static-card')).toHaveCount(6);
  expect(await loop.locator('.advantage-loop-message').evaluate(node => Number(getComputedStyle(node).opacity))).toBe(1);
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await expect(loop).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(1000);
  await expect(loop).toHaveAttribute('data-phase', 'sample');
  await expect(loop).toHaveAttribute('data-cycle', '0');
});

test('both method pipelines remain complete on a phone without JavaScript', async ({ browser }) => {
  const page = await browser.newPage({ javaScriptEnabled: false, viewport: { width: 360, height: 1000 } });
  try {
    await page.goto('http://127.0.0.1:4173/');
    const loop = page.locator('[data-advantage-loop]');
    await expect(loop).toHaveAttribute('data-state', 'static');
    await expect(loop.locator('.advantage-static-card')).toHaveCount(6);
    await expect(loop.locator('.advantage-static-card [data-final-advantage]')).toHaveText(['+1.20', '+1.00', '+0.80', '−1.20', '−1.00', '−0.80']);
    expect(await loop.locator('.advantage-static').textContent()).not.toMatch(/\b[uw]\s*=/);
    const sessions = await loop.locator('.advantage-static-card').evaluateAll(nodes => nodes.map(node =>
      node.querySelectorAll('.advantage-session-block').length));
    expect(sessions).toEqual([2, 4, 8, 2, 4, 8]);
    await expect(page.locator('[data-session-shaping] .shaping-type-group')).toHaveCount(3);
    expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  } finally { await page.close(); }
});

test('without IntersectionObserver the static trajectory pipeline is preserved', async ({ page }) => {
  await page.addInitScript(() => { delete window.IntersectionObserver; });
  await page.goto('/');
  const loop = page.locator('[data-advantage-loop]');
  await expect(loop).toHaveAttribute('data-state', 'static');
  await expect(loop.locator('.advantage-static-card')).toHaveCount(6);
});
