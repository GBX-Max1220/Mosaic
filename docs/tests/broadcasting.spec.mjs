import { test, expect } from '@playwright/test';

for (const width of [360, 390, 600, 768, 1440]) {
  test(`broadcasting draws tokens below separate session cards at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto('/');
    const diagram = page.locator('.broadcast-example');
    const groups = diagram.locator('.broadcast-group');
    await expect(groups).toHaveCount(3, { timeout: 1000 });
    await expect(diagram).toHaveAccessibleName(/trajectory advantage.*sessions.*trainable tokens/i);
    const advantage = await diagram.locator('.signal-value').textContent();
    for (const group of await groups.all()) {
      const session = group.locator('.broadcast-session');
      await expect(session.locator('i')).toHaveText(advantage);
      await expect(session.locator('.broadcast-token')).toHaveCount(0);
      const tokens = group.locator('.broadcast-token');
      await expect(tokens).toHaveCount(3);
      await expect(tokens.locator(':scope > span')).toHaveText(['t1', 't2', 'tn']);
      for (const token of await tokens.all()) await expect(token.locator('i')).toHaveText(advantage);
      const geometry = await group.evaluate(node => {
        const group = node.getBoundingClientRect();
        const session = node.querySelector('.broadcast-session').getBoundingClientRect();
        const tokens = [...node.querySelectorAll('.broadcast-token')].map(token => {
          const rect = token.getBoundingClientRect();
          return { rect, clipped: [...token.children].some(child => {
            const text = child.getBoundingClientRect();
            return text.left < rect.left || text.right > rect.right;
          }) };
        });
        return { group, session, tokens, border: getComputedStyle(node).borderTopWidth };
      });
      expect(geometry.session.width).toBeLessThanOrEqual(120);
      expect(geometry.session.height).toBeLessThanOrEqual(85);
      expect(geometry.border).toBe('0px');
      for (const { rect, clipped } of geometry.tokens) {
        expect(rect.top).toBeGreaterThan(geometry.session.bottom + 30);
        expect(rect.width).toBeLessThan(geometry.session.width);
        expect(rect.left).toBeGreaterThanOrEqual(geometry.group.left);
        expect(rect.right).toBeLessThanOrEqual(geometry.group.right);
        expect(clipped).toBe(false);
      }
    }
    const layout = await groups.evaluateAll(nodes => ({
      groups: nodes.map(node => node.getBoundingClientRect()),
      sessions: nodes.map(node => node.querySelector('.broadcast-session').getBoundingClientRect()),
      tokens: nodes.map(node => node.querySelector('.broadcast-token').getBoundingClientRect()),
      diagram: nodes[0].closest('.broadcast-example').getBoundingClientRect(),
      prose: nodes[0].closest('.prose').getBoundingClientRect(),
      overflow: document.documentElement.scrollWidth > innerWidth,
    }));
    expect(layout.diagram.width).toBeCloseTo(layout.prose.width, 0);
    expect(layout.overflow).toBe(false);
    if (width < 600) {
      expect(layout.groups[0].bottom).toBeLessThan(layout.groups[1].top);
      expect(layout.groups[1].bottom).toBeLessThan(layout.groups[2].top);
    } else {
      expect(layout.sessions[0].top).toBeCloseTo(layout.sessions[2].top, 0);
      expect(layout.tokens[0].top).toBeCloseTo(layout.tokens[2].top, 0);
      expect(layout.tokens[0].top).toBeGreaterThan(Math.max(...layout.sessions.map(session => session.bottom)));
    }
    await expect(diagram).not.toContainText(/loss mask|The same advantage is applied/i);
  });
}

test('the independent token level remains complete without JavaScript', async ({ browser }) => {
  const page = await browser.newPage({ javaScriptEnabled: false, viewport: { width: 360, height: 1000 } });
  try {
    await page.goto('http://127.0.0.1:4173/');
    const diagram = page.locator('.broadcast-example');
    await diagram.scrollIntoViewIfNeeded();
    await expect(diagram.locator('.broadcast-token')).toHaveCount(9, { timeout: 1000 });
    await expect(diagram.locator('.broadcast-session .broadcast-token')).toHaveCount(0);
    expect(await diagram.evaluate(node => node.scrollWidth <= node.clientWidth)).toBe(true);
  } finally { await page.close(); }
});
