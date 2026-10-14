import { test, expect } from '@playwright/test';

async function startLoop(page) {
  const now = new Date('2026-10-11T00:00:00Z');
  await page.clock.install({ time: now });
  await page.clock.pauseAt(now);
  await page.goto('/');
  const figure = page.locator('[data-session-loop]');
  await expect(figure).toHaveCount(1);
  await figure.scrollIntoViewIfNeeded();
  await expect(figure).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(800);
  return figure;
}

async function shownTokens(figure) {
  return figure.locator('.session-chip').evaluateAll(tokens => tokens
    .filter(token => Number(getComputedStyle(token).opacity) > .9)
    .map(token => token.dataset.role));
}

for (const width of [360, 390, 768, 1440]) {
  test(`single-session loop matches the reading column without overflow at ${width}px`, async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');
    const figure = page.locator('[data-session-loop]');
    await expect(figure).toHaveCount(1);
    const layout = await figure.evaluate(element => {
      const rect = element.getBoundingClientRect();
      const prose = element.closest('.prose').getBoundingClientRect();
      return {
        width: rect.width, proseWidth: prose.width, left: rect.left,
        documentWidth: document.documentElement.scrollWidth,
        before: element.previousElementSibling.tagName,
        after: element.nextElementSibling.tagName,
        followingDemonstrations: element.nextElementSibling.nextElementSibling.querySelectorAll('[data-multi-session]').length,
      };
    });
    expect(layout.width).toBeCloseTo(layout.proseWidth, 0);
    expect(layout.left).toBeGreaterThanOrEqual(20);
    expect(layout.documentWidth).toBeLessThanOrEqual(width);
    expect(layout.before).toBe('P');
    expect(layout.after).toBe('P');
    expect(layout.followingDemonstrations).toBe(2);
    await expect(figure).toHaveAttribute('data-state', 'static');
    await expect(figure.locator('.session-static')).toBeVisible();
  });
}

test('single-session rollout grows its prefix, stays continuous, becomes masked and loops', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const figure = await startLoop(page);
  expect(await shownTokens(figure)).toEqual(['sys']);
  const initialWidth = await figure.locator('.session-prefix').evaluate(node => node.getBoundingClientRect().width);

  await page.clock.runFor(3000);
  expect(await shownTokens(figure)).toEqual(['sys', 'action']);
  await page.clock.runFor(1500);
  expect(await shownTokens(figure)).toEqual(['sys', 'action', 'observation']);
  const longerWidth = await figure.locator('.session-prefix').evaluate(node => node.getBoundingClientRect().width);
  expect(longerWidth).toBeGreaterThan(initialWidth);

  await page.clock.runFor(3500);
  expect(await shownTokens(figure)).toEqual(['sys', 'action', 'observation', 'action', 'observation']);
  expect(await figure.locator('.session-context').evaluate(node => Number(getComputedStyle(node).opacity))).toBeGreaterThan(.9);
  await page.clock.runFor(9600);
  await expect(figure).toHaveAttribute('data-phase', 'result');
  expect((await shownTokens(figure)).length).toBe(16);
  await expect(figure.locator('.session-chip[data-role="sys"]')).toHaveCount(1);
  expect(await figure.locator('.session-chip[data-role="sys"] .session-chip-label').evaluate(node => Number(getComputedStyle(node).opacity))).toBe(0);
  await expect(figure.locator('.session-chip[data-final] .session-chip-hint')).toHaveText('Final answer');

  await page.clock.runFor(6600);
  await expect(figure).toHaveAttribute('data-phase', 'training');
  const masks = await figure.locator('.session-mask').evaluateAll(nodes => nodes.map(node => ({
    mask: node.textContent, opacity: Number(getComputedStyle(node).opacity),
  })));
  expect(masks.map(mask => mask.mask)).toEqual(['0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1']);
  expect(masks.every(mask => mask.opacity > .9)).toBe(true);
  expect((await shownTokens(figure)).length).toBe(16);
  await expect(figure.locator('.session-loop-message')).toContainText('loss-masked training segment');

  await page.clock.runFor(6800);
  await expect(figure).toHaveAttribute('data-phase', 'update');
  expect(await figure.locator('.session-trainer').evaluate(node => Number(getComputedStyle(node).opacity))).toBe(0);
  await page.clock.runFor(8000);
  await expect(figure).toHaveAttribute('data-cycle', '1');
  await expect(figure).toHaveAttribute('data-phase', 'system');
  expect(await shownTokens(figure)).toEqual(['sys']);
  expect(errors).toEqual([]);
});

test('merged explanations remain readable throughout every stage of the 38-second loop', async ({ page }) => {
  const figure = await startLoop(page);
  const message = figure.locator('.session-loop-message');
  const stages = [
    [0, 3000, 'The system prompt starts a new session.'],
    [3000, 8000, 'An action is generated from the current prefix; its observation is appended next.'],
    [8000, 17000, 'Each new interaction extends the same prefix, in the same context window.'],
    [17000, 21000, 'The session ends with the final action returning the task result, completing one continuous history.'],
    [21000, 31000, 'This completed session becomes one loss-masked training segment, which is then passed to the trainer.'],
    [31000, 35000, 'One rollout. One session. One training segment.'],
    [35000, 38000, 'The demonstration starts again with a fresh system prompt.'],
  ];
  let elapsed = 800;
  for (const [start, end, text] of stages) {
    await page.clock.runFor(start + 1000 - elapsed);
    await expect(message).toHaveText(text);
    await page.clock.runFor(end - 200 - (start + 1000));
    await expect(message).toHaveText(text);
    await expect(figure).toHaveAttribute('data-cycle', '0');
    elapsed = end - 200;
  }
  await page.clock.runFor(1000);
  await expect(figure).toHaveAttribute('data-cycle', '1');
  await expect(message).toHaveText('The system prompt starts a new session.');
});

test('explanations fade out and in at every stage boundary, including the loop seam', async ({ page }) => {
  const figure = await startLoop(page);
  const message = figure.locator('.session-loop-message');
  const opacity = () => message.evaluate(node => Number(getComputedStyle(node).opacity));
  let elapsed = 800;
  for (const boundary of [3000, 8000, 17000, 21000, 31000, 35000, 38000]) {
    await page.clock.runFor(boundary - 450 - elapsed);
    const previous = await message.textContent();
    expect(await opacity()).toBe(1);
    await page.clock.runFor(300);
    expect(await message.textContent()).toBe(previous);
    expect(await opacity()).toBeGreaterThan(.1);
    expect(await opacity()).toBeLessThan(.8);
    await page.clock.runFor(150);
    expect(await opacity()).toBeLessThan(.02);
    // The clock boundary can fall between rAF ticks; allow the next frame.
    await page.clock.runFor(16);
    expect(await message.textContent()).not.toBe(previous);
    await page.clock.runFor(134);
    expect(await opacity()).toBeGreaterThan(.1);
    expect(await opacity()).toBeLessThan(.8);
    await page.clock.runFor(300);
    expect(await opacity()).toBe(1);
    elapsed = boundary + 450;
  }
  await expect(figure).toHaveAttribute('data-cycle', '1');
  await expect(message).toHaveText('The system prompt starts a new session.');
});

for (const width of [360, 1440]) {
  test(`summary centers and the three-second restart returns its subtitle to the top at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const figure = await startLoop(page);
    const position = () => figure.evaluate(element => {
      const card = element.getBoundingClientRect();
      const message = element.querySelector('.session-loop-message').getBoundingClientRect();
      return {
        x: message.left + message.width / 2 - (card.left + card.width / 2),
        y: message.top + message.height / 2 - (card.top + card.height / 2),
        top: message.top - card.top,
        left: message.left - card.left, right: message.right - card.left,
        width: card.width,
      };
    });
    const initial = await position();
    await page.clock.runFor(31000); // 31800: summary has finished moving.
    const summary = await position();
    expect(Math.abs(summary.x)).toBeLessThan(1);
    expect(Math.abs(summary.y)).toBeLessThan(1);
    expect(await figure.locator('.session-trainer').evaluate(node => Number(getComputedStyle(node).opacity))).toBe(0);
    await page.clock.runFor(4700); // 36500: restart sentence stays in the center.
    await expect(figure.locator('.session-loop-message')).toHaveText('The demonstration starts again with a fresh system prompt.');
    const restart = await position();
    expect(Math.abs(restart.x)).toBeLessThan(1);
    expect(Math.abs(restart.y)).toBeLessThan(1);
    await page.clock.runFor(1000); // 37500: midway through returning to the top.
    const returning = await position();
    expect(returning.top).toBeGreaterThan(initial.top + 10);
    expect(returning.top).toBeLessThan(restart.top - 10);
    expect(returning.left).toBeGreaterThanOrEqual(0);
    expect(returning.right).toBeLessThanOrEqual(returning.width);
    await page.clock.runFor(516); // 38016: the first frame after the loop boundary.
    await expect(figure).toHaveAttribute('data-cycle', '1');
    await expect(figure.locator('.session-loop-message')).toHaveText('The system prompt starts a new session.');
    const next = await position();
    expect(next.top).toBeCloseTo(initial.top, 0);
    expect(next.left).toBeCloseTo(initial.left, 0);
  });
}

test('prefix bracket stays hidden until a new card appears, including after looping', async ({ page }) => {
  const figure = await startLoop(page);
  const opacity = () => figure.locator('.session-prefix').evaluate(node => Number(getComputedStyle(node).opacity));
  expect(await shownTokens(figure)).toEqual(['sys']);
  expect(await opacity()).toBe(0);
  await page.clock.runFor(2000); // 2800ms: still only SYS.
  expect(await opacity()).toBe(0);
  await page.clock.runFor(500); // 3300ms: an action appears, outside the prefix.
  expect(await opacity()).toBeGreaterThan(.9);
  await page.clock.runFor(36500); // 39800ms: next cycle begins with SYS only.
  await expect(figure).toHaveAttribute('data-cycle', '1');
  expect(await shownTokens(figure)).toEqual(['sys']);
  expect(await opacity()).toBe(0);
});

for (const width of [360, 1440]) {
  test(`loss mask label aligns with digits without overlap, even during delivery at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const figure = await startLoop(page);
    await page.clock.runFor(24200); // 25000ms: the mask is fully visible.
    for (const advance of [0, 4700]) {
      await page.clock.runFor(advance);
      const geometry = await figure.evaluate(element => {
        const heading = element.querySelector('.session-mask-heading');
        const label = heading.getBoundingClientRect();
        const digits = [...element.querySelectorAll('.session-mask')].map(node => {
          const rect = node.getBoundingClientRect();
          return { centerY: rect.top + rect.height / 2, left: rect.left, opacity: Number(getComputedStyle(node).opacity) };
        });
        return {
          centerY: label.top + label.height / 2, right: label.right,
          opacity: Number(getComputedStyle(heading).opacity), digits,
        };
      });
      expect(geometry.opacity).toBeGreaterThan(.1);
      expect(geometry.digits.every(digit => Math.abs(digit.centerY - geometry.centerY) < .5)).toBe(true);
      expect(geometry.digits[0].left).toBeGreaterThanOrEqual(geometry.right + 2);
      expect(geometry.digits.every(digit => digit.opacity > .1)).toBe(true);
    }
  });
}

test('prefix excludes a generating card and absorbs it only after a short hold', async ({ page }) => {
  const figure = await startLoop(page);
  const right = locator => locator.evaluate(node => node.getBoundingClientRect().right);
  const prefix = figure.locator('.session-prefix');
  const sys = figure.locator('.session-chip').nth(0);
  const action = figure.locator('.session-chip').nth(1);
  await page.clock.runFor(2400); // 3200ms: the action is still being generated.
  expect(await right(prefix)).toBeCloseTo(await right(sys), 0);
  await page.clock.runFor(350); // 3550ms: generated, but still in the hold period.
  expect(await action.evaluate(node => Number(getComputedStyle(node).opacity))).toBe(1);
  expect(await right(prefix)).toBeCloseTo(await right(sys), 0);
  await page.clock.runFor(200); // 3750ms: the prefix now expands toward the action.
  expect(await right(prefix)).toBeGreaterThan(await right(sys));
  expect(await right(prefix)).toBeLessThan(await right(action));
  await page.clock.runFor(250);
  expect(await right(prefix)).toBeCloseTo(await right(action), 0);

  const observation = figure.locator('.session-chip').nth(2);
  await page.clock.runFor(700); // 4700ms: the observation is outside the prefix.
  expect(await right(prefix)).toBeCloseTo(await right(action), 0);
  await page.clock.runFor(850);
  expect(await right(prefix)).toBeCloseTo(await right(observation), 0);
});

test('the minimal animation keeps changing explanations and in-scene hints only', async ({ page }) => {
  const figure = await startLoop(page);
  await expect(figure.locator('.session-loop-message')).toBeVisible();
  await expect(figure.locator('.session-loop-message')).toContainText('system prompt');
  await expect(figure.locator('.session-loop-heading, .session-loop-step, .session-loop-legend, figcaption')).toHaveCount(0);
  const hint = figure.locator('.session-chip[data-role="sys"] .session-chip-hint');
  await expect(hint).toHaveText('System prompt');
  expect(await hint.evaluate(node => Number(getComputedStyle(node).opacity))).toBe(1);
  await page.clock.runFor(3000);
  await expect(figure.locator('.session-loop-message')).toContainText('An action');
  await expect(figure.locator('.session-chip').nth(1).locator('.session-chip-hint')).toHaveText('Action');
});

test('single-session loop pauses offscreen and in the background without skipping its timeline', async ({ page }) => {
  const figure = await startLoop(page);
  await page.locator('#takeaways').scrollIntoViewIfNeeded();
  await expect(figure).toHaveAttribute('data-state', 'paused');
  const pausedPhase = await figure.getAttribute('data-phase');
  await page.clock.runFor(10000);
  await expect(figure).toHaveAttribute('data-phase', pausedPhase);
  await expect(figure).toHaveAttribute('data-cycle', '0');
  await figure.scrollIntoViewIfNeeded();
  await expect(figure).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(4500);
  expect(await shownTokens(figure)).toEqual(['sys', 'action', 'observation']);

  await page.evaluate(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, value: true });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(figure).toHaveAttribute('data-state', 'paused');
  const prefix = await figure.locator('.session-prefix').getAttribute('style');
  await page.clock.runFor(10000);
  expect(await figure.locator('.session-prefix').getAttribute('style')).toBe(prefix);
  await page.evaluate(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, value: false });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await expect(figure).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(3500);
  expect(await shownTokens(figure)).toEqual(['sys', 'action', 'observation', 'action', 'observation']);
});

test('live training tokens and masks remain inside the scene after resizing to mobile', async ({ page }) => {
  const figure = await startLoop(page);
  await page.clock.runFor(24200);
  await expect(figure).toHaveAttribute('data-phase', 'training');
  for (const width of [360, 390, 768, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await figure.scrollIntoViewIfNeeded();
    await expect(figure).toHaveAttribute('data-state', 'playing');
    await page.clock.runFor(32);
    const geometry = await figure.evaluate(element => {
      const scene = element.querySelector('.session-loop-scene').getBoundingClientRect();
      const boxes = [...element.querySelectorAll('.session-chip, .session-mask')].map(node => {
        const rect = node.getBoundingClientRect();
        return { left: rect.left, right: rect.right };
      });
      return {
        left: scene.left, right: scene.right, boxes,
        maskFont: parseFloat(getComputedStyle(element.querySelector('.session-mask')).fontSize),
        documentWidth: document.documentElement.scrollWidth,
      };
    });
    expect(geometry.boxes).toHaveLength(32);
    expect(geometry.boxes.every(box => box.left >= geometry.left - 1 && box.right <= geometry.right + 1)).toBe(true);
    expect(geometry.maskFont).toBeGreaterThanOrEqual(12);
    expect(geometry.documentWidth).toBeLessThanOrEqual(width);
    await expect(figure).toHaveAttribute('data-phase', 'training');
  }
});

test('single-session loop switches to a complete static explanation for reduced motion', async ({ page }) => {
  const figure = await startLoop(page);
  await page.clock.runFor(31000); // Switch while the animated subtitle is centered.
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(figure).toHaveAttribute('data-state', 'static');
  await expect(figure.locator('.session-static')).toBeVisible();
  await expect(figure.locator('.session-live')).toBeHidden();
  await expect(figure.locator('.session-static')).toContainText('One training segment');
  await expect(figure.locator('.session-static')).toContainText('RL Trainer');
  const subtitle = await figure.locator('.session-loop-message').evaluate(node => ({
    transform: getComputedStyle(node).transform,
    opacity: Number(getComputedStyle(node).opacity),
    top: node.getBoundingClientRect().top - node.closest('figure').getBoundingClientRect().top,
  }));
  expect(subtitle.transform).toBe('none');
  expect(subtitle.opacity).toBe(1);
  expect(subtitle.top).toBeLessThan(60);
  await page.clock.runFor(30000);
  await expect(figure).toHaveAttribute('data-state', 'static');
});

test('single-session explanation remains readable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 900 } });
  try {
    const page = await context.newPage();
    await page.goto('http://127.0.0.1:4173/');
    const figure = page.locator('[data-session-loop]');
    await expect(figure).toHaveCount(1);
    await expect(figure.locator('.session-static')).toBeVisible();
    await expect(figure).toContainText('Same context window');
    await expect(figure).toContainText('One training segment');
    await expect(figure).toHaveAccessibleName(/observation.*context/i);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  } finally { await context.close(); }
});
