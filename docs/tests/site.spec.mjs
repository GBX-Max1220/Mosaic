import { test, expect } from '@playwright/test';

test('readers can see the article, both experiment settings and pending resources', async ({ page }) => {
  const failures = [];
  page.on('response', response => {
    if (response.url().startsWith('http://127.0.0.1:4173/') && response.status() >= 400) failures.push(response.url());
  });
  await page.goto('/');
  await expect(page.locator('#paper-title')).toBeVisible();
  await expect(page.locator('main')).toContainText('38.86');
  await expect(page.locator('main')).toContainText('32.46');
  await expect(page.locator('#method')).toContainText('We propose Mosaic, a general method for multi-session agentic reinforcement learning');
  await expect(page.locator('#method figure')).toHaveCount(0);
  await expect(page.locator('[data-paper-authors]')).toBeVisible();
  for (const name of ['arXiv', 'Hugging Face Papers']) {
    await expect(page.getByRole('link', { name, exact: true })).toHaveAttribute('aria-disabled', 'true');
  }
  const code = page.getByRole('link', { name: 'Code', exact: true });
  await expect(code).toHaveAttribute('href', 'https://github.com/LARK-AI-Lab/Mosaic');
  await expect(code).not.toHaveAttribute('aria-disabled', 'true');
  await page.locator('#takeaways').scrollIntoViewIfNeeded();
  await expect(page.locator('img[data-paper-figure]')).toHaveCount(0);
  expect(failures).toEqual([]);
});

for (const width of [360, 1440]) {
  test(`the merged discussion keeps conclusions and four concise future directions without paper figures at ${width}px`, async ({ browser }) => {
    const context = await browser.newContext({ javaScriptEnabled: false, viewport: { width, height: 900 } });
    try {
      const page = await context.newPage();
      await page.goto('http://127.0.0.1:4173/');
      const discussion = page.locator('#takeaways');
      await expect(discussion.getByRole('heading', { level: 2 })).toHaveCount(1);
      await expect(discussion.getByRole('heading', { level: 3 })).toHaveCount(5);
      await expect(discussion.locator('.closing-note')).toBeVisible();
      await expect(discussion.locator('.citation')).toBeVisible();
      await expect(page.locator('main > .article-section')).toHaveCount(5);
      await expect(page.locator('main .paper-figure, main img[data-paper-figure]')).toHaveCount(0);
      const bounds = await discussion.locator('.prose').boundingBox();
      expect(bounds.width).toBeLessThanOrEqual(760);
      expect(bounds.x).toBeGreaterThanOrEqual(20);
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(width - 20);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    } finally {
      await context.close();
    }
  });
}

for (const width of [360, 390, 768, 1440]) {
  test(`article stays within the viewport at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');
    await expect(page.locator('#paper-title')).toBeVisible();
    const layout = await page.evaluate(() => ({
      viewport: window.innerWidth,
      document: document.documentElement.scrollWidth,
      article: document.querySelector('#overview .prose').getBoundingClientRect().width,
      figure: document.querySelector('.figure-wide').getBoundingClientRect().width,
      resultsProse: document.querySelector('#results .prose').getBoundingClientRect().toJSON(),
      resultsTable: document.querySelector('#results .results-table').getBoundingClientRect().toJSON(),
    }));
    expect(layout.document).toBeLessThanOrEqual(layout.viewport);
    expect(layout.article).toBeLessThanOrEqual(760);
    expect(layout.figure).toBeLessThanOrEqual(960);
    expect(layout.resultsTable.width).toBeCloseTo(layout.resultsProse.width, 1);
    expect(layout.resultsTable.left).toBeCloseTo(layout.resultsProse.left, 1);
    expect(layout.resultsTable.right).toBeCloseTo(layout.resultsProse.right, 1);
  });
}

for (const width of [1024, 1280, 1440]) {
  test(`the full paper title fits one line without widening the article at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');
    const layout = await page.locator('#paper-title').evaluate(title => {
      const range = document.createRange();
      range.selectNodeContents(title);
      const rects = Array.from(range.getClientRects());
      return {
        lines: new Set(rects.map(rect => Math.round(rect.top))).size,
        textLeft: Math.min(...rects.map(rect => rect.left)),
        textRight: Math.max(...rects.map(rect => rect.right)),
        fontSize: parseFloat(getComputedStyle(title).fontSize),
        heroWidth: title.closest('.hero').getBoundingClientRect().width,
        articleWidth: document.querySelector('#overview .prose').getBoundingClientRect().width,
        viewport: innerWidth,
      };
    });
    expect(layout.lines).toBe(1);
    expect(layout.textLeft).toBeGreaterThanOrEqual(20);
    expect(layout.textRight).toBeLessThanOrEqual(layout.viewport - 20);
    expect(layout.fontSize).toBeGreaterThanOrEqual(40);
    expect(layout.heroWidth).toBeGreaterThan(layout.articleWidth);
    expect(layout.heroWidth).toBeLessThanOrEqual(1200);
    expect(layout.articleWidth).toBeLessThanOrEqual(760);
  });
}

test('the title wraps at a readable size on a phone without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 900 });
  await page.goto('/');
  const layout = await page.locator('#paper-title').evaluate(title => {
    const range = document.createRange();
    range.selectNodeContents(title);
    return {
      lines: new Set(Array.from(range.getClientRects(), rect => Math.round(rect.top))).size,
      fontSize: parseFloat(getComputedStyle(title).fontSize),
      documentWidth: document.documentElement.scrollWidth,
    };
  });
  expect(layout.lines).toBeGreaterThan(1);
  expect(layout.fontSize).toBeGreaterThanOrEqual(30);
  expect(layout.documentWidth).toBeLessThanOrEqual(360);
});

test('the complete article and diagrams are readable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4173/');
  await expect(page.locator('#paper-title')).toBeVisible();
  await expect(page.locator('[data-definition-loop]')).toHaveAttribute('data-state', 'static');
  await page.locator('#results').scrollIntoViewIfNeeded();
  await expect(page.getByRole('cell', { name: '38.86%', exact: true })).toBeVisible();
  await context.close();
});

test('real metadata replaces pending text, wraps and enables valid resource links', async ({ page }) => {
  await page.route('**/static/data/paper.json', route => route.fulfill({
    json: {
      authors: ['First Researcher With A Long Name', 'Second Researcher', 'Third Researcher', 'Fourth Researcher'],
      affiliations: ['An Institute With A Long Name', 'Another University'],
      links: { arxiv: 'https://arxiv.org/abs/1234.56789', code: 'https://github.com/example/mosaic', huggingface: 'https://huggingface.co/papers/1234.56789' },
      siteUrl: 'https://example.org/mosaic/',
    },
  }));
  await page.setViewportSize({ width: 360, height: 800 });
  await page.goto('/');
  await expect(page.locator('[data-paper-authors]')).toContainText('First Researcher With A Long Name');
  await expect(page.locator('[data-paper-affiliations]')).toHaveText('An Institute With A Long Name · Another University');
  await expect(page.getByRole('link', { name: 'arXiv', exact: true })).toHaveAttribute('href', 'https://arxiv.org/abs/1234.56789');
  await expect(page.getByRole('link', { name: 'Code', exact: true })).toHaveAttribute('href', 'https://github.com/example/mosaic');
  await expect(page.getByRole('link', { name: 'Hugging Face Papers', exact: true })).toHaveAttribute('href', 'https://huggingface.co/papers/1234.56789');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute('href', 'https://example.org/mosaic/');
});

test('unsafe links stay disabled and markup in author metadata remains plain text', async ({ page }) => {
  await page.route('**/static/data/paper.json', route => route.fulfill({
    json: { authors: ['<img src=x onerror=alert(1)>'], affiliations: [], links: { arxiv: 'javascript:alert(1)', code: 'http://example.org', huggingface: '' } },
  }));
  await page.goto('/');
  await expect(page.locator('[data-paper-authors]')).toHaveText('<img src=x onerror=alert(1)>');
  await expect(page.locator('[data-paper-authors] img')).toHaveCount(0);
  await expect(page.locator('[data-resource][href]')).toHaveCount(0);
});

test('configuration load failure preserves the readable HTML metadata', async ({ page, browser }) => {
  const fallbackPage = await browser.newPage({ javaScriptEnabled: false });
  await fallbackPage.goto('http://127.0.0.1:4173/');
  const fallbackAuthors = await fallbackPage.locator('[data-paper-authors]').innerText();
  await fallbackPage.close();
  await page.route('**/static/data/paper.json', route => route.abort());
  await page.goto('/');
  await expect(page.locator('[data-paper-authors]')).toHaveText(fallbackAuthors);
  await expect(page.locator('[data-resource][href]')).toHaveCount(0);
  await expect(page.locator('#takeaways')).toContainText('session utility and allocation');
});

test('all assets and configuration resolve under a project subpath', async ({ page }) => {
  const requests = [];
  await page.route('http://127.0.0.1:4173/project/**', async route => {
    requests.push(new URL(route.request().url()).pathname);
    const response = await route.fetch({ url: route.request().url().replace('/project/', '/') });
    await route.fulfill({ response });
  });
  await page.goto('/project/');
  await expect(page.locator('[data-definition-loop]')).toHaveAttribute('data-state', 'ready');
  await expect.poll(() => requests.includes('/project/static/data/paper.json')).toBe(true);
  expect(requests).toContain('/project/static/css/style.css');
  expect(requests).toContain('/project/static/js/main.js');
  expect(requests).toContain('/project/static/js/multi-session-loop.js');
  expect(requests).toContain('/project/static/js/definition-loop.js');
  expect(requests).toContain('/project/static/js/advantage-loop.js');
  await page.locator('#forking-animation').scrollIntoViewIfNeeded();
  await expect(page.locator('#forking-animation')).toHaveAttribute('data-state', 'playing');
  await page.locator('[data-definition-loop]').scrollIntoViewIfNeeded();
  await expect(page.locator('[data-definition-loop]')).toHaveAttribute('data-state', 'playing');
  await expect(page.locator('#takeaways .closing-note')).toBeAttached();
});
