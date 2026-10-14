import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}

function httpsUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && !url.username && !url.password ? url : null;
  } catch {
    return null;
  }
}

function names(value) {
  return Array.isArray(value) ? value.filter(item => typeof item === 'string' && item.trim()).map(item => item.trim()) : [];
}

/** Assemble only public files into the disposable _site directory. */
export async function preparePages(sourceRoot, { siteUrl = '' } = {}) {
  // Read required inputs before replacing a prior generated artifact.
  let html = await readFile(join(sourceRoot, 'index.html'), 'utf8');
  const paper = JSON.parse(await readFile(join(sourceRoot, 'static/data/paper.json'), 'utf8'));
  const output = join(sourceRoot, '_site');
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  await cp(join(sourceRoot, 'static'), join(output, 'static'), { recursive: true });
  await cp(join(sourceRoot, '.nojekyll'), join(output, '.nojekyll'));

  for (const [attribute, value, separator] of [
    ['data-paper-authors', names(paper.authors), ', '],
    ['data-paper-affiliations', names(paper.affiliations), ' · '],
  ]) {
    if (value.length) {
      html = html.replace(new RegExp(`(<p\\b[^>]*${attribute}[^>]*>)[\\s\\S]*?(</p>)`), (_, open, close) => open + escapeHtml(value.join(separator)) + close);
    }
  }

  html = html.replace(/<a\b([^>]*\bdata-resource="([^"]+)"[^>]*)>/g, (original, attributes, resource) => {
    const url = httpsUrl(paper.links?.[resource]);
    if (!url) return original;
    const enabled = attributes.replace(/\saria-disabled="true"|\stabindex="-1"/g, '');
    return `<a${enabled} href="${escapeHtml(url.href)}">`;
  });

  const site = httpsUrl(paper.siteUrl) || httpsUrl(siteUrl);
  if (site) {
    if (!site.pathname.endsWith('/')) site.pathname += '/';
    site.search = '';
    site.hash = '';
    paper.siteUrl = site.href;
    const image = new URL('static/images/social-preview.png', site).href;
    const metadata = [
      `<link rel="canonical" href="${escapeHtml(site.href)}">`,
      `<meta property="og:url" content="${escapeHtml(site.href)}">`,
      `<meta property="og:image" content="${escapeHtml(image)}">`,
      '<meta property="og:image:width" content="1200">',
      '<meta property="og:image:height" content="630">',
      '<meta property="og:image:alt" content="Mosaic: Multi-Session Agentic Reinforcement Learning">',
      `<meta name="twitter:image" content="${escapeHtml(image)}">`,
    ].join('\n  ');
    html = html.replace('</head>', `  ${metadata}\n</head>`);
  }

  await writeFile(join(output, 'index.html'), html);
  await writeFile(join(output, 'static/data/paper.json'), JSON.stringify(paper, null, 2) + '\n');
  return output;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const root = fileURLToPath(new URL('../', import.meta.url));
  console.log(`Prepared public site: ${await preparePages(root, { siteUrl: process.env.PAGES_BASE_URL })}`);
}
