/* Complete static diagrams are the baseline; JavaScript only adds a short reveal. */
(() => {
  'use strict';

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

  function setMeta(property, value) {
    let element = document.head.querySelector(`meta[property="${property}"]`);
    if (!element) {
      element = document.createElement('meta');
      element.setAttribute('property', property);
      document.head.append(element);
    }
    element.content = value;
  }

  function applyMetadata(paper) {
    if (!paper || typeof paper !== 'object') return;
    const authors = names(paper.authors);
    const affiliations = names(paper.affiliations);
    if (authors.length) document.querySelector('[data-paper-authors]').textContent = authors.join(', ');
    if (affiliations.length) document.querySelector('[data-paper-affiliations]').textContent = affiliations.join(' · ');

    for (const element of document.querySelectorAll('[data-resource]')) {
      const url = httpsUrl(paper.links?.[element.dataset.resource]);
      if (!url) continue;
      element.href = url.href;
      element.removeAttribute('aria-disabled');
      element.removeAttribute('tabindex');
    }

    const site = httpsUrl(paper.siteUrl);
    if (site) {
      let canonical = document.querySelector('link[rel="canonical"]');
      if (!canonical) {
        canonical = document.createElement('link');
        canonical.rel = 'canonical';
        document.head.append(canonical);
      }
      canonical.href = site.href;
      setMeta('og:url', site.href);
      const base = new URL(site.href);
      if (!base.pathname.endsWith('/')) base.pathname += '/';
      setMeta('og:image', new URL('static/images/social-preview.png', base).href);
      setMeta('og:image:width', '1200');
      setMeta('og:image:height', '630');
      setMeta('og:image:alt', 'Mosaic: Multi-Session Agentic Reinforcement Learning');
    }
  }

  function initializeAnimations() {
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (motion.matches || !('IntersectionObserver' in window)) return;
    const records = new Map();
    const duration = 4500;

    function finish(record) {
      clearTimeout(record.timer);
      record.timer = null;
      record.figure.dataset.state = 'complete';
      record.figure.removeAttribute('data-paused');
      observer.unobserve(record.figure);
    }

    function pause(record) {
      if (record.timer !== null) {
        record.remaining = Math.max(0, record.remaining - (performance.now() - record.startedAt));
        clearTimeout(record.timer);
        record.timer = null;
      }
      record.figure.dataset.paused = 'true';
    }

    function synchronize(record) {
      if (record.figure.dataset.state === 'complete') return;
      if (motion.matches) return finish(record);
      if (document.hidden || !record.inView) {
        if (record.figure.dataset.state === 'running') pause(record);
        return;
      }
      record.figure.dataset.state = 'running';
      record.figure.removeAttribute('data-paused');
      if (record.timer === null) {
        record.startedAt = performance.now();
        record.timer = setTimeout(() => finish(record), record.remaining);
      }
    }

    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        const record = records.get(entry.target);
        record.inView = entry.isIntersecting;
        synchronize(record);
      }
    }, { threshold: 0.15 });

    for (const figure of document.querySelectorAll('[data-animation]')) {
      for (const connector of figure.querySelectorAll('.connector')) connector.setAttribute('pathLength', '1');
      figure.dataset.state = 'ready';
      records.set(figure, { figure, remaining: duration, timer: null, startedAt: 0, inView: false });
      observer.observe(figure);
    }

    document.addEventListener('visibilitychange', () => records.forEach(synchronize));
    motion.addEventListener('change', () => records.forEach(synchronize));
  }

  initializeAnimations();
  fetch(new URL('static/data/paper.json', document.baseURI))
    .then(response => response.ok ? response.json() : null)
    .then(applyMetadata)
    .catch(() => { /* Keep the explicit pending metadata if configuration is unavailable. */ });
})();
