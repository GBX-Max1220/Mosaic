import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, readdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const module = await import('../scripts/prepare-pages.mjs').catch(() => null);

async function fixture(t, metadata) {
  const root = await mkdtemp(join(tmpdir(), 'mosaic-pages-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'static/data'), { recursive: true });
  await mkdir(join(root, 'static/css'), { recursive: true });
  await mkdir(join(root, '_site'), { recursive: true });
  await writeFile(join(root, 'index.html'), '<html><head><title>Mosaic</title></head><body><p data-paper-authors>Pending</p><p data-paper-affiliations>Pending</p><a role="link" aria-disabled="true" tabindex="-1" data-resource="code">Code</a></body></html>');
  await writeFile(join(root, '.nojekyll'), '');
  await writeFile(join(root, 'static/css/style.css'), 'body { color: black; }');
  await writeFile(join(root, 'static/data/paper.json'), JSON.stringify(metadata));
  await writeFile(join(root, 'paper.tex'), 'private manuscript source');
  await writeFile(join(root, '_site/stale.tex'), 'must not survive a rebuild');
  return root;
}

test('Pages package includes public assets and excludes manuscript and stale files', async t => {
  assert.equal(typeof module?.preparePages, 'function', 'The Pages packager has not been implemented');
  const root = await fixture(t, { authors: [], affiliations: [], links: {}, siteUrl: '' });
  const output = await module.preparePages(root);
  assert.deepEqual((await readdir(output)).sort(), ['.nojekyll', 'index.html', 'static']);
  assert.equal(await readFile(join(output, 'static/css/style.css'), 'utf8'), 'body { color: black; }');
  assert.equal(await readFile(join(root, 'paper.tex'), 'utf8'), 'private manuscript source');
});

test('packaged HTML contains safe real metadata for readers and social crawlers without JavaScript', async t => {
  assert.equal(typeof module?.preparePages, 'function', 'The Pages packager has not been implemented');
  const root = await fixture(t, { authors: ['Alice & Bob'], affiliations: ['Institute <A>'], links: { code: 'https://github.com/example/mosaic?x=1&y=2' }, siteUrl: '' });
  const output = await module.preparePages(root, { siteUrl: 'https://example.org/project' });
  const html = await readFile(join(output, 'index.html'), 'utf8');
  assert.ok(html.includes('Alice &amp; Bob'));
  assert.ok(html.includes('Institute &lt;A&gt;'));
  assert.ok(html.includes('href="https://github.com/example/mosaic?x=1&amp;y=2"'));
  assert.ok(!html.includes('aria-disabled="true"'));
  assert.ok(html.includes('property="og:image" content="https://example.org/project/static/images/social-preview.png"'));
  assert.ok(html.includes('rel="canonical" href="https://example.org/project/"'));
  const data = JSON.parse(await readFile(join(output, 'static/data/paper.json'), 'utf8'));
  assert.equal(data.siteUrl, 'https://example.org/project/');
  assert.equal(JSON.parse(await readFile(join(root, 'static/data/paper.json'), 'utf8')).siteUrl, '');
});

test('invalid publication links do not become live links in packaged HTML', async t => {
  assert.equal(typeof module?.preparePages, 'function', 'The Pages packager has not been implemented');
  const root = await fixture(t, { authors: [], affiliations: [], links: { code: 'javascript:alert(1)' }, siteUrl: '' });
  const output = await module.preparePages(root);
  const html = await readFile(join(output, 'index.html'), 'utf8');
  assert.ok(html.includes('aria-disabled="true"'));
  assert.ok(!html.includes('href='));
});
