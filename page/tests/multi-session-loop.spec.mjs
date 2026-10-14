import { test, expect } from '@playwright/test';

async function startLoop(page, mode, width = 1440) {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
  await page.setViewportSize({ width, height: 1000 });
  const now = new Date('2026-10-11T00:00:00Z');
  await page.clock.install({ time: now });
  await page.clock.pauseAt(now);
  await page.goto('/');
  const figure = page.locator(`[data-multi-session="${mode}"]`);
  await expect(figure).toHaveCount(1, { timeout: 1000 });
  expect(errors, 'page and asset errors before playback').toEqual([]);
  await figure.scrollIntoViewIfNeeded();
  await expect(figure).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(800);
  return figure;
}

const visibleChips = lane => lane.locator('.session-chip').evaluateAll(nodes => nodes
  .filter(node => Number(getComputedStyle(node).opacity) > .9)
  .map(node => node.querySelector('.session-chip-label').textContent));

test('compaction keeps the old history separate and seeds a third session with its summary', async ({ page }) => {
  const figure = await startLoop(page, 'compaction');
  const execution = figure.locator('[data-session-id="execution"]');
  const compaction = figure.locator('[data-session-id="compaction"]');
  const resumed = figure.locator('[data-session-id="resumed"]');
  expect(await visibleChips(execution)).toEqual(['SYS']);
  await page.clock.fastForward(7000); // 7800: the limit is detected before the first session closes.
  const oldHistory = await visibleChips(execution);
  expect(oldHistory).toEqual(['SYS', 'a', 'o', 'a', 'o', 'a', 'o']);
  await expect(figure).toHaveAttribute('data-phase', 'boundary');
  await page.clock.fastForward(7700); // 15500: dedicated summarization context.
  expect(await visibleChips(compaction)).toEqual(['SYS', 'History', 'Summary']);
  expect(await visibleChips(execution)).toEqual(oldHistory);
  expect(await visibleChips(resumed)).toEqual([]);
  await page.clock.fastForward(4000); // 19500: SYS + Summary, not SYS + the old action history.
  expect(await visibleChips(resumed)).toEqual(['SYS', 'Summary']);
  expect(await visibleChips(execution)).toEqual(oldHistory);
  await page.clock.fastForward(8300); // 27800: all three distinct contexts are visible.
  expect(await visibleChips(resumed)).toEqual(['SYS', 'Summary', 'a', 'o', 'a', 'o', 'a']);
  expect(await visibleChips(compaction)).toEqual(['SYS', 'History', 'Summary']);
  await expect(figure.locator('.multi-rollout')).toHaveCount(0);
  await expect(figure.locator('.multi-lane[data-status="closed"]')).toHaveCount(3);
});

test('forking returns only results while the main agent preserves one continuous session', async ({ page }) => {
  const figure = await startLoop(page, 'forking');
  const main = figure.locator('[data-session-id="main"]');
  const a = figure.locator('[data-session-id="subagent-a"]');
  const b = figure.locator('[data-session-id="subagent-b"]');
  const mainElement = await main.elementHandle();
  await page.clock.fastForward(7000); // 7800: main fork call; no child history in the parent.
  expect(await visibleChips(main)).toEqual(['SYS', 'a', 'o', 'a', 'o', 'Fork']);
  await page.clock.fastForward(10000); // 17800: both children have their own SYS, task and history.
  expect(await visibleChips(a)).toEqual(['SYS', 'Task A', 'a', 'o', 'a', 'o', 'a']);
  expect(await visibleChips(b)).toEqual(['SYS', 'Task B', 'a', 'o', 'a', 'o', 'a']);
  expect(await visibleChips(main)).toEqual(['SYS', 'a', 'o', 'a', 'o', 'Fork', 'o']);
  await page.clock.fastForward(5000); // 22800: the aggregate result is one observation.
  expect(await visibleChips(main)).toEqual(['SYS', 'a', 'o', 'a', 'o', 'Fork', 'o']);
  await expect(main.locator('.session-chip[data-kind="results"] .session-chip-hint')).toHaveText('Subagent results');
  expect(await main.evaluate((node, original) => node === original, mainElement)).toBe(true);
  await page.clock.fastForward(5000);
  expect(await visibleChips(main)).toEqual(['SYS', 'a', 'o', 'a', 'o', 'Fork', 'o', 'a', 'o', 'a']);
  await expect(main.locator('.session-chip[data-final] .session-chip-hint')).toHaveText('Final answer');
  await expect(figure.locator('.multi-rollout')).toHaveCount(0);
  await expect(figure.locator('.multi-lane[data-status="closed"]')).toHaveCount(3);
});

for (const width of [360, 1440]) {
  test(`forking return arrows always land on a visible observation inside the main session at ${width}px`, async ({ page }) => {
    const figure = await startLoop(page, 'forking', width);
    let previous = 800;
    for (const time of [18050, 18300, 18700, 19300, 20550, 20800, 23200, 24200, 25700, 26950]) {
      await page.clock.fastForward(time - previous);
      previous = time;
      const geometry = await figure.evaluate(node => {
        const main = node.querySelector('[data-session-id="main"]');
        const observation = main.querySelector('[data-kind="results"]');
        const card = observation.getBoundingClientRect();
        const frame = main.querySelector('.multi-lane-window').getBoundingClientRect();
        const endpoints = ['return-1', 'return-2'].flatMap(name => {
          const path = node.querySelector(`[data-link="${name}"]`);
          if (Number(getComputedStyle(path).opacity) <= .01) return [];
          const point = path.getPointAtLength(path.getTotalLength()).matrixTransform(path.getScreenCTM());
          return [{ x: point.x, y: point.y }];
        });
        return { opacity: Number(getComputedStyle(observation).opacity),
          card: { left: card.left, right: card.right, top: card.top, bottom: card.bottom, width: card.width },
          frame: { left: frame.left, right: frame.right, top: frame.top, bottom: frame.bottom }, endpoints };
      });
      expect(geometry.endpoints.length, `a result is returning at ${time}ms`).toBeGreaterThan(0);
      expect(geometry.opacity, `the receiving observation must already be visible at ${time}ms`).toBeGreaterThan(.99);
      expect(geometry.card.width).toBeGreaterThan(10);
      expect(geometry.card.left).toBeGreaterThanOrEqual(geometry.frame.left);
      expect(geometry.card.right).toBeLessThanOrEqual(geometry.frame.right);
      expect(geometry.card.top).toBeGreaterThan(geometry.frame.top);
      expect(geometry.card.bottom).toBeLessThan(geometry.frame.bottom);
      for (const point of geometry.endpoints) {
        expect(point.x).toBeCloseTo((geometry.card.left + geometry.card.right) / 2, 1);
        expect(point.y).toBeCloseTo(geometry.card.top, 1);
      }
    }
  });

  test(`forking connectors turn outside session frames and share vertical trunks at ${width}px`, async ({ page }) => {
    const figure = await startLoop(page, 'forking', width);
    const geometry = () => figure.evaluate(node => {
      const vertices = name => {
        const path = node.querySelector(`[data-link="${name}"]`);
        let x = 0;
        let y = 0;
        return Array.from(path.getAttribute('d').matchAll(/([MHV])([^MHV]*)/g), ([, command, coordinates]) => {
          const values = coordinates.trim().split(/[\s,]+/).map(Number);
          if (command === 'M') [x, y] = values;
          if (command === 'H') x = values[0];
          if (command === 'V') y = values[0];
          const point = new DOMPoint(x, y).matrixTransform(path.getScreenCTM());
          return { x: point.x, y: point.y };
        });
      };
      const frameTop = id => node.querySelector(`[data-session-id="${id}"] .multi-lane-window`).getBoundingClientRect().top;
      const tasks = ['subagent-a', 'subagent-b'].map(id => {
        const card = node.querySelector(`[data-session-id="${id}"] .session-chip[data-kind="task"]`).getBoundingClientRect();
        return { x: card.left + card.width / 2, y: card.top };
      });
      return {
        forks: [vertices('fork-1'), vertices('fork-2')],
        returns: [vertices('return-1'), vertices('return-2')],
        mainTop: frameTop('main'), childTops: ['subagent-a', 'subagent-b'].map(frameTop), tasks,
      };
    });
    await page.clock.fastForward(11700); // 12500: both delegated tasks are visible.
    const delegated = await geometry();
    delegated.forks.forEach((points, index) => {
      expect(points[1].y, 'first elbow must be above the main session frame').toBeLessThanOrEqual(delegated.mainTop - 6);
      expect(points[4].y, 'last elbow must be above the target session frame').toBeLessThanOrEqual(delegated.childTops[index] - 6);
      expect(points[5].x).toBeCloseTo(delegated.tasks[index].x, 1);
      expect(points[5].y, 'arrow must connect to the Task card').toBeCloseTo(delegated.tasks[index].y, 1);
    });
    expect(delegated.forks[0][2].x, 'delegation paths share one downward trunk').toBeCloseTo(delegated.forks[1][2].x, 1);
    await page.clock.fastForward(10300); // 22800: both results have returned.
    const returned = await geometry();
    expect(returned.returns[0][1].x, 'return paths share one upward trunk').toBeCloseTo(returned.returns[1][1].x, 1);
    returned.returns.forEach(points => expect(points[2].y).toBeLessThanOrEqual(returned.mainTop - 6));
  });
}

for (const mode of ['compaction', 'forking']) {
  for (const width of [360, 1440]) {
    test(`${mode} concludes with compact highlighted sessions before a separate centered restart at ${width}px`, async ({ page }) => {
      const figure = await startLoop(page, mode, width);
      const message = figure.locator('.session-loop-message');
      const snapshot = () => figure.evaluate(node => {
        const scene = node.querySelector('.multi-session-scene');
        const message = node.querySelector('.session-loop-message');
        const card = node.getBoundingClientRect();
        const caption = message.getBoundingClientRect();
        const opacity = element => {
          let value = 1;
          for (let ancestor = element; ancestor && ancestor !== node; ancestor = ancestor.parentElement) {
            value *= Number(getComputedStyle(ancestor).opacity);
          }
          return value;
        };
        return {
          sceneOpacity: Number(getComputedStyle(scene).opacity), captionOpacity: opacity(message),
          captionTop: caption.top - card.top,
          captionX: caption.left + caption.width / 2 - card.left - card.width / 2,
          captionY: caption.top + caption.height / 2 - card.top - card.height / 2,
          frames: [...node.querySelectorAll('.multi-lane-window')].map(frame => {
            const rect = frame.getBoundingClientRect();
            const style = getComputedStyle(frame);
            return { top: rect.top, bottom: rect.bottom, center: rect.left + rect.width / 2,
              opacity: opacity(frame), color: style.borderColor, background: style.backgroundColor, border: style.borderStyle };
          }),
          auxiliaryOpacities: [...node.querySelectorAll('.multi-links, .multi-lane-label, .session-chip-hint, .multi-context-budget, .multi-transfer, .multi-rollout')].map(opacity),
        };
      });
      const initial = await snapshot();
      await page.clock.fastForward(26000); // 26800: the histories are complete, before the closing overview.
      const before = await snapshot();
      await page.clock.fastForward(2200); // 29000: only the three highlighted sessions and top caption remain.
      const overview = await snapshot();
      await expect(message).toHaveText('One rollout branches into multiple sessions, each with its own interaction history.');
      expect(overview.captionTop).toBeCloseTo(initial.captionTop, 0);
      expect(overview.captionOpacity).toBe(1);
      expect(overview.sceneOpacity).toBe(1);
      expect(Math.max(...overview.auxiliaryOpacities)).toBe(0);
      const oldHeight = before.frames[2].bottom - before.frames[0].top;
      const newHeight = overview.frames[2].bottom - overview.frames[0].top;
      expect(newHeight, 'the three sessions visibly gather into a compact stack').toBeLessThan(oldHeight * .75);
      overview.frames.forEach((frame, index) => {
        expect(frame.opacity).toBe(1);
        expect(frame.color).not.toBe(before.frames[index].color);
        expect(frame.background).not.toBe('rgba(0, 0, 0, 0)');
        expect(frame.border).toBe('solid');
        expect(frame.center).toBeCloseTo(overview.frames[0].center, 0);
        if (index > 0) expect(frame.top - overview.frames[index - 1].bottom).toBeGreaterThanOrEqual(16);
      });
      await page.clock.fastForward(5250); // 34250: caption and sessions fade out together.
      const fading = await snapshot();
      expect(fading.sceneOpacity).toBeGreaterThan(0);
      expect(fading.sceneOpacity).toBeLessThan(1);
      expect(fading.captionOpacity).toBeCloseTo(fading.sceneOpacity, 2);
      await page.clock.fastForward(450); // 34700: the old scene is gone before restart text appears.
      expect((await snapshot()).sceneOpacity).toBe(0);
      expect((await snapshot()).captionOpacity).toBe(0);
      await page.clock.fastForward(450); // 35150: restart fades in at the center, without the old scene.
      await expect(message).toHaveText('The demonstration starts again with a fresh system prompt.');
      const restarting = await snapshot();
      expect(restarting.sceneOpacity).toBe(0);
      expect(restarting.captionOpacity).toBeGreaterThan(0);
      expect(restarting.captionOpacity).toBeLessThan(1);
      expect(Math.abs(restarting.captionX)).toBeLessThan(1);
      expect(Math.abs(restarting.captionY)).toBeLessThan(1);
      await page.clock.fastForward(650); // 35800: restart is fully readable, still centered.
      expect((await snapshot()).captionOpacity).toBe(1);
      await page.clock.fastForward(3000); // 38800: a fresh rollout restores the normal annotations and unhighlighted frames.
      await expect(figure).toHaveAttribute('data-phase', 'initial');
      const fresh = await snapshot();
      expect(fresh.captionTop).toBeCloseTo(initial.captionTop, 0);
      expect(fresh.frames[0].color).toBe(initial.frames[0].color);
      expect(await figure.locator('.multi-lane-label').first().evaluate(node => Number(getComputedStyle(node).opacity))).toBe(1);
    });
  }

  test(`${mode} shows completion in its header without prefix brackets and clears it on restart`, async ({ page }) => {
    const figure = await startLoop(page, mode, 360);
    await expect(figure.locator('.multi-prefix')).toHaveCount(0);
    const labels = figure.locator('.multi-lane-label');
    const initial = await labels.allTextContents();
    expect(initial.every(label => !label.includes('(Completed)'))).toBe(true);
    await page.clock.fastForward(8400); // 9200: only the execution session has ended in compaction.
    await expect(labels.first()).toHaveText(initial[0] + (mode === 'compaction' ? ' (Completed)' : ''));
    await page.clock.fastForward(18600); // 27800: all three sessions have ended.
    await expect(labels).toHaveText(initial.map(label => `${label} (Completed)`));
    const overlaps = await figure.locator('.multi-lane').evaluateAll(nodes => nodes.filter(node => {
      const label = node.querySelector('.multi-lane-label').getBoundingClientRect();
      const frame = node.querySelector('.multi-lane-window').getBoundingClientRect();
      return label.bottom > frame.top - 4;
    }).map(node => node.dataset.sessionId));
    expect(overlaps).toEqual([]);
    await page.clock.fastForward(11000); // 38800: a fresh loop removes every Completed suffix.
    await expect(labels).toHaveText(initial);
    await expect(figure.locator('.multi-prefix')).toHaveCount(0);
  });

  test(`${mode} captions fade, center only for restart, return and loop from SYS`, async ({ page }) => {
    const figure = await startLoop(page, mode, 360);
    const message = figure.locator('.session-loop-message');
    const geometry = () => figure.evaluate(node => {
      const card = node.getBoundingClientRect();
      const text = node.querySelector('.session-loop-message').getBoundingClientRect();
      return { x: text.left + text.width / 2 - (card.left + card.width / 2),
        y: text.top + text.height / 2 - (card.top + card.height / 2), top: text.top - card.top };
    });
    const initialTop = (await geometry()).top;
    await page.clock.runFor(3050); // 3850: an unchanged explanation is fully readable.
    expect(await message.evaluate(node => Number(getComputedStyle(node).opacity))).toBe(1);
    const boundary = mode === 'compaction' ? 7000 : 6000;
    await page.clock.fastForward(boundary - 150 - 3850);
    expect(await message.evaluate(node => Number(getComputedStyle(node).opacity))).toBeLessThan(.8);
    await page.clock.fastForward(35800 - (boundary - 150));
    const centered = await geometry();
    expect(Math.abs(centered.x)).toBeLessThan(1);
    expect(Math.abs(centered.y)).toBeLessThan(1);
    await expect(message).toHaveText('The demonstration starts again with a fresh system prompt.');
    for (const resizedWidth of [768, 360]) {
      await page.setViewportSize({ width: resizedWidth, height: 1000 });
      await figure.scrollIntoViewIfNeeded();
      await page.clock.runFor(32);
      const resized = await geometry();
      expect(Math.abs(resized.x)).toBeLessThan(1);
      expect(Math.abs(resized.y)).toBeLessThan(1);
    }
    await page.clock.fastForward(636); // 36500, including the two 32ms resize frames: restart stays centered.
    await expect(message).toHaveText('The demonstration starts again with a fresh system prompt.');
    expect(Math.abs((await geometry()).y)).toBeLessThan(1);
    await page.clock.fastForward(1000); // 37500: smoothly moving back.
    expect((await geometry()).top).toBeGreaterThan(initialTop + 10);
    await page.clock.fastForward(1300); // 38800: fresh SYS, no previous branches or summary.
    await expect(figure).toHaveAttribute('data-cycle', '1');
    await expect(figure).toHaveAttribute('data-phase', 'initial');
    expect((await geometry()).top).toBeCloseTo(initialTop, 0);
    expect(await visibleChips(figure.locator('.multi-lane').first())).toEqual(['SYS']);
    expect(await figure.locator('.multi-lane').nth(1).evaluate(node => Number(getComputedStyle(node).opacity))).toBe(0);
  });

  test(`${mode} pauses its caption and diagram offscreen and in the background`, async ({ page }) => {
    const figure = await startLoop(page, mode);
    await page.clock.runFor(3050);
    await page.locator('#takeaways').scrollIntoViewIfNeeded();
    await expect(figure).toHaveAttribute('data-state', 'paused');
    const state = () => figure.evaluate(node => node.innerHTML);
    const paused = await state();
    await page.clock.runFor(10000);
    expect(await state()).toBe(paused);
    await figure.scrollIntoViewIfNeeded();
    await expect(figure).toHaveAttribute('data-state', 'playing');
    await page.clock.runFor(500);
    expect(await state()).not.toBe(paused);
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { configurable: true, value: true });
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await expect(figure).toHaveAttribute('data-state', 'paused');
    const hidden = await state();
    await page.clock.runFor(10000);
    expect(await state()).toBe(hidden);
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { configurable: true, value: false });
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await page.clock.runFor(500);
    expect(await state()).not.toBe(hidden);
  });

  test(`${mode} live cards and annotations fit on a phone throughout all phases`, async ({ page }) => {
    const figure = await startLoop(page, mode, 360);
    let elapsed = 800;
    for (const moment of [2800, 7800, 12500, 15500, 19500, 22800, 27800, 31800, 37500, 38800]) {
      await page.clock.fastForward(moment - elapsed);
      const overflow = await figure.evaluate(node => {
        const scene = node.querySelector('.multi-session-scene').getBoundingClientRect();
        const card = node.getBoundingClientRect();
        const chips = [...node.querySelectorAll('.session-live .session-chip')].filter(chip =>
          Number(chip.style.opacity) > .9 && Number(chip.closest('.multi-lane')?.style.opacity ?? 1) > .1);
        return { badChips: chips.filter(chip => {
          const rect = chip.getBoundingClientRect();
          return rect.left < scene.left - 1 || rect.right > scene.right + 1 || rect.top < scene.top - 1 || rect.bottom > scene.bottom + 1;
        }).length,
        badText: [...node.querySelectorAll('.session-loop-message, .multi-lane-label')].filter(text => {
          const rect = text.getBoundingClientRect();
          return rect.left < card.left - 1 || rect.right > card.right + 1;
        }).length, documentWidth: document.documentElement.scrollWidth };
      });
      expect(overflow.badChips, `chips at ${moment}ms`).toBe(0);
      expect(overflow.badText, `annotations at ${moment}ms`).toBe(0);
      expect(overflow.documentWidth).toBeLessThanOrEqual(360);
      elapsed = moment;
    }
  });
}

for (const width of [360, 390, 768, 1024, 1440]) {
  test(`inference cards share a wide desktop row or stack within the reading column at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/');
    for (const mode of ['compaction', 'forking']) {
      const figure = page.locator(`[data-multi-session="${mode}"]`);
      await expect(figure).toHaveCount(1, { timeout: 1000 });
      await expect(figure).toHaveAttribute('data-state', 'static');
      await expect(figure.locator('.session-static')).toBeVisible();
      await expect(figure.locator('.session-live')).toBeHidden();
      await expect(figure.locator('.multi-static-rollout')).toContainText('One rollout · 3 sessions');
      const sizes = await figure.evaluate(node => ({
        card: node.getBoundingClientRect().width, prose: node.closest('.prose').getBoundingClientRect().width,
        documentWidth: document.documentElement.scrollWidth,
      }));
      if (width < 1000) expect(sizes.card).toBeCloseTo(sizes.prose, 0);
      else expect(sizes.card).toBeLessThan(sizes.prose);
      expect(sizes.documentWidth).toBeLessThanOrEqual(width);
    }
    const layout = await page.locator('#overview').evaluate(node => {
      const compaction = node.querySelector('#compaction-animation').getBoundingClientRect();
      const forking = node.querySelector('#forking-animation').getBoundingClientRect();
      const prose = node.querySelector('.prose').getBoundingClientRect();
      return { compaction, forking, prose };
    });
    if (width >= 1000) {
      expect(layout.compaction.top).toBeCloseTo(layout.forking.top, 0);
      expect(layout.compaction.height).toBeCloseTo(layout.forking.height, 0);
      expect(layout.compaction.right).toBeLessThan(layout.forking.left);
      const combinedWidth = layout.forking.right - layout.compaction.left;
      expect(combinedWidth).toBeGreaterThan(layout.prose.width);
      expect(combinedWidth).toBeLessThanOrEqual(1200);
      expect((layout.compaction.left + layout.forking.right) / 2).toBeCloseTo(width / 2, 0);
    } else {
      expect(layout.compaction.bottom).toBeLessThan(layout.forking.top);
      expect(layout.compaction.left).toBeCloseTo(layout.forking.left, 0);
    }
    await expect(page.locator('[data-animation="sessions"]')).toHaveCount(0);
    const order = await page.locator('#overview figure').evaluateAll(nodes => nodes.map(node => node.id));
    expect(order).toEqual(['single-session-animation', 'compaction-animation', 'forking-animation']);
  });
}

for (const width of [360, 1440]) {
  test(`the multi-session definition follows both demonstrations within the reading column at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/');
    const definition = page.locator('.setting-definition');
    await expect(definition).toHaveCount(1, { timeout: 1000 });
    await expect(definition).toHaveAccessibleName('Multi-Session Agentic Reinforcement Learning');
    const bounds = await definition.evaluate(node => ({
      card: node.getBoundingClientRect(),
      prose: node.closest('.prose').getBoundingClientRect(),
      previousFigure: document.querySelector('#forking-animation').getBoundingClientRect(),
      contentWidth: node.scrollWidth, clientWidth: node.clientWidth,
    }));
    expect(bounds.card.width).toBeCloseTo(bounds.prose.width, 0);
    expect(bounds.card.top).toBeGreaterThan(bounds.previousFigure.bottom);
    expect(bounds.contentWidth).toBeLessThanOrEqual(bounds.clientWidth);
    await expect(definition.getByRole('math')).toHaveCount(0);
    await expect(definition.getByRole('img')).toHaveAccessibleName(/rollout.*sessions.*training segment.*policy update/i);
  });
}

for (const mode of ['compaction', 'forking']) {
  for (const width of [360, 1440]) {
    test(`${mode} session frames center their cards with generous padding at ${width}px`, async ({ page }) => {
      const figure = await startLoop(page, mode, width);
      let elapsed = 800;
      for (const moment of [800, 2800, 4200, 7800, 12500, 15500, 19500, 22800, 27800]) {
        await page.clock.fastForward(moment - elapsed);
        const padding = await figure.locator('.multi-lane').evaluateAll(lanes => lanes.flatMap(lane => {
          if (Number(lane.style.opacity) < .1) return [];
          const chips = [...lane.querySelectorAll('.session-chip')].filter(chip => Number(chip.style.opacity) > .05);
          if (!chips.length) return [];
          const frame = lane.querySelector('.multi-lane-window').getBoundingClientRect();
          const rects = chips.map(chip => chip.getBoundingClientRect());
          return [{ id: lane.dataset.sessionId,
            left: Math.min(...rects.map(rect => rect.left)) - frame.left,
            right: frame.right - Math.max(...rects.map(rect => rect.right)),
            top: Math.min(...rects.map(rect => rect.top)) - frame.top,
            bottom: frame.bottom - Math.max(...rects.map(rect => rect.bottom)) }];
        }));
        for (const row of padding) {
          expect(row.left, `${row.id} left at ${moment}`).toBeGreaterThanOrEqual(15);
          expect(row.right, `${row.id} right at ${moment}`).toBeGreaterThanOrEqual(15);
          expect(row.top, `${row.id} top at ${moment}`).toBeGreaterThanOrEqual(11);
          expect(row.bottom, `${row.id} bottom at ${moment}`).toBeGreaterThanOrEqual(11);
          expect(row.left, `${row.id} horizontal centering at ${moment}`).toBeCloseTo(row.right, 0);
          expect(row.top, `${row.id} vertical centering at ${moment}`).toBeCloseTo(row.bottom, 0);
        }
        elapsed = moment;
      }
    });
  }
}

for (const width of [360, 1440]) {
  test(`compaction handoff arrows avoid visible labels at ${width}px`, async ({ page }) => {
    const figure = await startLoop(page, 'compaction', width);
    let elapsed = 800;
    for (const moment of [11800, 12500, 15500, 17800, 19500, 22800]) {
      await page.clock.fastForward(moment - elapsed);
      const collisions = await figure.evaluate(root => {
        const textRects = [...root.querySelectorAll('.multi-lane-label, .session-chip-hint, .session-chip-label')]
          .filter(text => {
            for (let ancestor = text; ancestor && ancestor !== root; ancestor = ancestor.parentElement) {
              if (Number(getComputedStyle(ancestor).opacity) <= .1) return false;
            }
            return true;
          }).map(text => {
            const range = document.createRange();
            range.selectNodeContents(text);
            return { text: text.textContent, rect: range.getBoundingClientRect() };
          });
        return [...root.querySelectorAll('.multi-links [data-link]')].filter(path => Number(path.style.opacity) > .1)
          .flatMap(path => {
            const transform = path.getScreenCTM();
            const length = path.getTotalLength();
            const points = Array.from({ length: Math.ceil(length / 2) + 1 }, (_, index) =>
              path.getPointAtLength(Math.min(length, index * 2)).matrixTransform(transform));
            return textRects.filter(({ rect }) => points.some(point => point.x > rect.left - 1 && point.x < rect.right + 1 && point.y > rect.top - 1 && point.y < rect.bottom + 1))
              .map(({ text }) => `${path.dataset.link}: ${text}`);
          });
      });
      expect(collisions, `arrows at ${moment}`).toEqual([]);
      elapsed = moment;
    }
  });

  test(`compaction arrows attach to visible endpoints and disappear after handoff at ${width}px`, async ({ page }) => {
    const figure = await startLoop(page, 'compaction', width);
    const endpoints = name => figure.evaluate((root, name) => {
      const path = root.querySelector(`[data-link="${name}"]`);
      const sourceNode = name === 'history'
        ? root.querySelector('[data-session-id="execution"] .multi-lane-window')
        : root.querySelector('[data-session-id="compaction"] [data-kind="summary"]');
      const targetNode = root.querySelector(`[data-session-id="${name === 'history' ? 'compaction' : 'resumed'}"] .multi-lane-window`);
      const source = sourceNode.getBoundingClientRect();
      const target = targetNode.getBoundingClientRect();
      const transform = path.getScreenCTM();
      const start = path.getPointAtLength(0).matrixTransform(transform);
      const end = path.getPointAtLength(path.getTotalLength()).matrixTransform(transform);
      const maxX = Math.max(...Array.from({ length: 21 }, (_, index) =>
        path.getPointAtLength(path.getTotalLength() * index / 20).matrixTransform(transform).x));
      return { opacity: Number(path.style.opacity),
        sourceVisible: Number(sourceNode.closest('.multi-lane').style.opacity) * Number(sourceNode.style.opacity || 1),
        targetVisible: Number(targetNode.closest('.multi-lane').style.opacity),
        detour: maxX - Math.max(source.right, target.right),
        sourceGap: Math.hypot(start.x - source.right, start.y - source.top - source.height / 2),
        targetGap: Math.hypot(end.x - target.right, end.y - target.top - target.height / 2) };
    }, name);
    await page.clock.fastForward(11700); // 12500: history handoff into a visible compaction context.
    for (const [key, value] of Object.entries(await endpoints('history'))) {
      if (key.endsWith('Gap')) expect(value, key).toBeLessThan(1);
      else if (key === 'detour') expect(value, key).toBeLessThan(40);
      else expect(value, key).toBeGreaterThan(key === 'sourceVisible' ? .5 : .9);
    }
    await page.clock.fastForward(3000); // 15500: no leftover history wiring.
    expect(await figure.locator('[data-link="history"]').evaluate(node => Number(node.style.opacity))).toBe(0);
    await page.clock.fastForward(2300); // 17800: SYS is visible, the Summary input is not yet visible.
    for (const [key, value] of Object.entries(await endpoints('summary'))) {
      if (key.endsWith('Gap')) expect(value, key).toBeLessThan(1);
      else if (key === 'detour') expect(value, key).toBeLessThan(40);
      else expect(value, key).toBeGreaterThan(key === 'sourceVisible' ? .5 : .9);
    }
    await page.clock.fastForward(1700); // 19500: input delivered; both handoff arrows have faded out.
    expect(await figure.locator('.multi-links [data-link]').evaluateAll(nodes => nodes.every(node => Number(node.style.opacity) === 0))).toBe(true);
  });
}

test('compaction detects a visible context limit before the execution session closes', async ({ page }) => {
  const figure = await startLoop(page, 'compaction', 360);
  const execution = figure.locator('[data-session-id="execution"]');
  const limit = figure.locator('.multi-limit');
  await page.clock.fastForward(5800); // 6600: filling, but no detected boundary yet.
  await expect(execution).toHaveAttribute('data-status', 'active');
  expect(await limit.evaluate(node => Number(node.style.opacity))).toBe(0);
  await page.clock.fastForward(800); // 7400: detection is distinct from ending.
  await expect(execution).toHaveAttribute('data-status', 'active');
  await expect(limit).toHaveText('Context limit detected');
  const appearance = await limit.evaluate(node => ({ opacity: Number(node.style.opacity), font: parseFloat(getComputedStyle(node).fontSize) }));
  expect(appearance.opacity).toBe(1);
  expect(appearance.font).toBeGreaterThanOrEqual(12);
  await expect(execution.locator('.multi-lane-window')).toHaveAttribute('data-context-limit', 'true');
  expect(await execution.locator('.multi-budget-fill').evaluate(node => node.style.width)).toBe('100%');
  const budgetHintGap = await execution.evaluate(root => {
    const meter = root.querySelector('.multi-context-budget').getBoundingClientRect();
    const hint = [...root.querySelectorAll('.session-chip-hint')].at(-1).getBoundingClientRect();
    return hint.top - meter.bottom;
  });
  expect(budgetHintGap).toBeGreaterThanOrEqual(4);
  const gap = await figure.evaluate(root => {
    const message = root.querySelector('.session-loop-message').getBoundingClientRect();
    const badge = root.querySelector('.multi-limit').getBoundingClientRect();
    return badge.top - message.bottom;
  });
  expect(gap).toBeGreaterThanOrEqual(4);
  await page.clock.fastForward(1800); // 9200: the session ends after detection.
  await expect(execution).toHaveAttribute('data-status', 'closed');
  await expect(limit).toHaveText('Context limit · session ended');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(limit).toBeHidden();
});

test('both desktop inference loops advance together and keep their session histories after a layout resize', async ({ page }) => {
  const compaction = await startLoop(page, 'compaction');
  const forking = page.locator('#forking-animation');
  await expect(forking).toHaveAttribute('data-state', 'playing', { timeout: 1000 });
  await page.clock.fastForward(14700);
  await expect(compaction).toHaveAttribute('data-phase', 'compaction');
  await expect(forking).toHaveAttribute('data-phase', 'branches');
  for (const width of [768, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    await compaction.scrollIntoViewIfNeeded();
    await expect(compaction).toHaveAttribute('data-state', 'playing');
    await page.clock.runFor(32);
    await expect(compaction).toHaveAttribute('data-cycle', '0');
    await expect(compaction).toHaveAttribute('data-phase', 'compaction');
    const overflow = await compaction.locator('.session-loop-message').evaluate(node => {
      const text = node.getBoundingClientRect();
      const stage = node.closest('.session-loop-stage').getBoundingClientRect();
      return { left: text.left - stage.left, right: text.right - stage.right, bottom: text.bottom - stage.bottom };
    });
    expect(overflow.left).toBeGreaterThanOrEqual(-1);
    expect(overflow.right).toBeLessThanOrEqual(1);
    expect(overflow.bottom).toBeLessThanOrEqual(1);
  }
  await expect(forking).toHaveAttribute('data-state', 'playing');
});

test('new inference demonstrations remain complete without JavaScript', async ({ browser }) => {
  const page = await browser.newPage({ javaScriptEnabled: false, viewport: { width: 360, height: 1000 } });
  try {
    await page.goto('http://127.0.0.1:4173/');
    for (const mode of ['compaction', 'forking']) {
      const figure = page.locator(`[data-multi-session="${mode}"]`);
      await expect(figure).toHaveCount(1, { timeout: 1000 });
      await expect(figure.locator('.session-static')).toBeVisible();
      await expect(figure.locator('.multi-static-row')).toHaveCount(3);
      expect((await figure.locator('.multi-static-row > p').allTextContents()).every(label => label.endsWith(' (Completed)'))).toBe(true);
      await expect(figure).toHaveAccessibleName(/one rollout.*three.*session/i);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  } finally { await page.close(); }
});

test('switching to reduced motion clears an already centered caption and resumes a fresh loop', async ({ page }) => {
  const figure = await startLoop(page, 'compaction');
  // This case checks a media switch, not every intermediate animation frame.
  await page.clock.fastForward(35000);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(figure).toHaveAttribute('data-state', 'static');
  const caption = figure.locator('.session-loop-message');
  expect(await caption.evaluate(node => getComputedStyle(node).transform)).toBe('none');
  expect(await caption.evaluate(node => Number(getComputedStyle(node).opacity))).toBe(1);
  await expect(figure.locator('.multi-static-rollout')).toBeVisible();
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await figure.scrollIntoViewIfNeeded();
  await expect(figure).toHaveAttribute('data-state', 'playing');
  await page.clock.runFor(800);
  await expect(figure).toHaveAttribute('data-phase', 'initial');
  await expect(figure).toHaveAttribute('data-cycle', '0');
  expect(await visibleChips(figure.locator('.multi-lane').first())).toEqual(['SYS']);
});

test('generated summary annotations stay below their frame and above the next session', async ({ page }) => {
  const figure = await startLoop(page, 'compaction');
  await page.clock.fastForward(14700); // 15500: the summary and its hint are fully visible.
  const lane = figure.locator('[data-session-id="compaction"]');
  const geometry = await lane.evaluate(node => {
    const hint = node.querySelector('[data-kind="summary"] .session-chip-hint').getBoundingClientRect();
    const frame = node.querySelector('.multi-lane-window').getBoundingClientRect();
    const next = node.nextElementSibling.querySelector('.multi-lane-label').getBoundingClientRect();
    return { hintTop: hint.top, hintBottom: hint.bottom, frameBottom: frame.bottom, nextLabelTop: next.top };
  });
  expect(geometry.hintTop).toBeGreaterThanOrEqual(geometry.frameBottom + 4);
  expect(geometry.hintBottom).toBeLessThanOrEqual(geometry.nextLabelTop - 8);
});

for (const width of [360, 1440]) {
  test(`returning Result B stays within the card and does not cross Subagent A's history at ${width}px`, async ({ page }) => {
    const figure = await startLoop(page, 'forking', width);
    await page.clock.fastForward(18500); // 19300: Result B is midway through its handoff.
    const geometry = await figure.evaluate(node => {
      const packet = [...node.querySelectorAll('.multi-transfer')].find(item => item.textContent === 'Result B');
      const rect = packet.getBoundingClientRect();
      const scene = node.querySelector('.multi-session-scene').getBoundingClientRect();
      const otherCards = [...node.querySelectorAll('[data-session-id="subagent-a"] .session-chip')].map(chip => chip.getBoundingClientRect());
      return { opacity: Number(getComputedStyle(packet).opacity), left: rect.left, right: rect.right,
        sceneLeft: scene.left, sceneRight: scene.right,
        overlaps: otherCards.some(card => rect.left < card.right && rect.right > card.left && rect.top < card.bottom && rect.bottom > card.top) };
    });
    expect(geometry.opacity).toBeGreaterThan(.5);
    expect(geometry.left).toBeGreaterThanOrEqual(geometry.sceneLeft);
    expect(geometry.right).toBeLessThanOrEqual(geometry.sceneRight);
    expect(geometry.overlaps).toBe(false);
  });
}
