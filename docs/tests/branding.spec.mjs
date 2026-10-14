import { test, expect } from '@playwright/test';

async function paintedGrid(group) {
  return group.locator('rect').evaluateAll(rects => rects.map(rect => {
    const box = rect.getBBox();
    return { x: box.x, y: box.y, width: box.width, height: box.height,
      fill: getComputedStyle(rect).fill, opacity: Number(getComputedStyle(rect).opacity) };
  }));
}

function expectMosaic(cells) {
  expect(cells).toHaveLength(9);
  expect(new Set(cells.map(cell => cell.x)).size).toBe(3);
  expect(new Set(cells.map(cell => cell.y)).size).toBe(3);
  expect(new Set(cells.map(cell => cell.fill)).size).toBeGreaterThanOrEqual(4);
  for (const cell of cells) {
    expect(cell.width).toBe(cell.height);
    expect(cell.width).toBeGreaterThan(0);
    expect(cell.fill).not.toBe('none');
    expect(cell.opacity).toBe(1);
  }
  for (const axis of ['x', 'y']) {
    const positions = [...new Set(cells.map(cell => cell[axis]))].sort((a, b) => a - b);
    expect(positions[1] - positions[0]).toBeGreaterThan(cells[0].width);
    expect(positions[2] - positions[1]).toBeCloseTo(positions[1] - positions[0]);
  }
}

for (const width of [360, 1440]) {
  test(`the brand paints nine distinct tiles without crowding navigation at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');
    const brand = page.getByRole('link', { name: 'Mosaic, back to top' });
    await expect(brand).toBeVisible();
    expectMosaic(await paintedGrid(brand.locator('svg')));
    const layout = await brand.evaluate(node => ({
      brandRight: node.getBoundingClientRect().right,
      navLeft: document.querySelector('.site-header nav').getBoundingClientRect().left,
      icon: node.querySelector('svg').getBoundingClientRect().width,
      overflow: document.documentElement.scrollWidth > innerWidth,
    }));
    expect(layout.brandRight).toBeLessThan(layout.navLeft);
    expect(layout.icon).toBeGreaterThanOrEqual(22);
    expect(layout.overflow).toBe(false);
    await page.locator('#overview').scrollIntoViewIfNeeded();
    await brand.click();
    await expect(page).toHaveURL(/#top$/);
  });
}

for (const asset of ['favicon.svg', 'social-preview.svg']) {
  test(`${asset} paints the same heatmap as the page header`, async ({ page }) => {
    await page.goto('/');
    const header = await paintedGrid(page.locator('.wordmark svg'));
    await page.goto(`/static/images/${asset}`);
    const assetCells = await paintedGrid(page.locator('svg > g').first());
    expectMosaic(assetCells);
    expect(assetCells.map(cell => cell.fill)).toEqual(header.map(cell => cell.fill));
    // Compare painted cell layout independent of the enclosing SVG's scale.
    const normalized = cells => cells.map(cell => ({
      x: (cell.x - cells[0].x) / cells[0].width,
      y: (cell.y - cells[0].y) / cells[0].height,
      width: cell.width / cells[0].width,
    }));
    expect(normalized(assetCells)).toEqual(normalized(header));
  });
}

test('the published PNG preview contains the updated heatmap, not a stale icon', async ({ page }) => {
  await page.goto('/');
  const header = await paintedGrid(page.locator('.wordmark svg'));
  const raster = await page.evaluate(async () => {
    const image = new Image();
    image.src = '/static/images/social-preview.png';
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    // These are cell centers in the 1200 × 630 sharing image, not SVG edges.
    const fills = [78, 92, 106].flatMap(y => [84, 98, 112].map(x => {
      const [r, g, b] = context.getImageData(x, y, 1, 1).data;
      return `rgb(${r}, ${g}, ${b})`;
    }));
    return { width: image.naturalWidth, height: image.naturalHeight, fills };
  });
  expect(raster.width).toBe(1200);
  expect(raster.height).toBe(630);
  expect(raster.fills).toEqual(header.map(cell => cell.fill));
});
