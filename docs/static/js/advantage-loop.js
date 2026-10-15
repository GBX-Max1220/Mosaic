/* A mixed-outcome prompt group: rank session counts within reward subgroups,
 * then use positive weights to modulate the original outcome advantage.
 * The complete HTML example is also the source of the animated rollouts.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-advantage-loop]');
  if (!root || !('IntersectionObserver' in window)) return;
  const scene = root.querySelector('.advantage-live');
  const message = root.querySelector('.advantage-loop-message');
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const duration = 30000;
  const rho = .2;
  const sessionEntrance = 350;
  const timing = { counted: 10000, scoring: 13000, grouping: 16000, grouped: 18000,
    advantage: 19000, preference: 20500, modulation: 22000, reset: 27000 };
  const stages = [
    [0, 'sample', 'Generate six rollouts in parallel. Each small block is one session.'],
    [6500, 'waiting', 'The group must wait for the longest rollout to finish.'],
    [timing.counted, 'counted', 'All rollouts are complete. N is the number of sessions in each rollout.'],
    [timing.scoring, 'scoring', 'Score each rollout: green for success, red for failure.'],
    [timing.grouping, 'groups', 'Split by outcome and sort each subgroup from fewest to most sessions.'],
    [timing.advantage, 'advantage', 'Start from the outcome advantage. Favor fewer sessions after success, more after failure.'],
    [timing.modulation, 'modulation', 'Adjust each trajectory’s advantage while preserving its outcome sign.'],
    [timing.reset, 'reset', 'Start again with a new group of rollouts.'],
  ];
  const ease = value => {
    const t = Math.max(0, Math.min(1, value));
    return t * t * (3 - 2 * t);
  };
  const mix = (a, b, t) => a + (b - a) * t;
  const signed = value => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(2)}`;
  const make = (tag, className, value) => {
    const node = document.createElement(tag);
    node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  };
  const specs = [...root.querySelectorAll('.advantage-static-card')].map(node => ({
    id: Number(node.dataset.rolloutId), outcome: Number(node.dataset.outcome), count: Number(node.dataset.sessions),
  })).sort((a, b) => a.id - b.id);
  if (!specs.length) return;
  const mean = specs.reduce((sum, spec) => sum + spec.outcome, 0) / specs.length;
  const deviation = Math.sqrt(specs.reduce((sum, spec) => sum + (spec.outcome - mean) ** 2, 0) / specs.length);
  const subgroups = [0, 1].map(outcome => specs.filter(spec => spec.outcome === outcome));
  const groupTop = 84;
  const groupCardHeight = 72;
  const groupStride = 86;
  const arrowHeight = (Math.max(...subgroups.map(group => group.length)) - 1) * groupStride + groupCardHeight;
  const paces = specs.map((_, index) => [1, 1.05, .9, 1.2, .9, 1.2][index % 6]);
  // Different rollout speeds leave a visible tail. Normalize the schedule so
  // the last session finishes before counts and scoring become available.
  const paceUnit = 8400 / Math.max(1, ...specs.map((spec, index) => (spec.count - 1) * paces[index]));
  const maxSessions = Math.max(...specs.map(spec => spec.count));
  const sampleOrder = [...specs].sort((a, b) => a.count - b.count || b.outcome - a.outcome || a.id - b.id);

  const prompt = make('div', 'advantage-live-label', `One prompt · ${specs.length} rollouts`);
  const labels = [1, 0].map(outcome => {
    const node = make('div', 'advantage-live-label', outcome ? 'Success' : 'Failure');
    node.dataset.outcome = String(outcome);
    return node;
  });
  const arrows = [1, 0].map(outcome => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    node.classList.add('advantage-direction');
    node.dataset.outcome = String(outcome);
    node.dataset.direction = outcome ? 'up' : 'down';
    node.setAttribute('viewBox', `0 0 18 ${arrowHeight}`);
    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    arrow.setAttribute('points', `9,0 18,20 12,20 12,${arrowHeight} 6,${arrowHeight} 6,20 0,20`);
    if (!outcome) arrow.setAttribute('transform', `rotate(180 9 ${arrowHeight / 2})`);
    node.append(arrow);
    return node;
  });
  scene.append(prompt, ...labels, ...arrows);
  const cards = specs.map((spec, index) => {
    const group = subgroups[spec.outcome];
    const sameCount = group.filter(item => item.count === spec.count).length;
    const rankPosition = group.filter(item => item.count < spec.count).length + (sameCount - 1) / 2;
    const tied = group.every(item => item.count === group[0].count);
    const rank = group.length > 1 && !tied ? 2 * rankPosition / (group.length - 1) - 1 : 0;
    const weight = Number((1 - rho * rank).toFixed(6));
    const base = deviation ? (spec.outcome - mean) / deviation : 0;
    // The visible example is mixed. Retain the paper's conservative fallbacks
    // if its HTML source is later changed to an all-success/all-failure group.
    const final = deviation ? weight * base : mean === 1 ? -.2 * rank : .1 * Math.min(rank, 0);
    const node = make('div', 'advantage-live-card');
    Object.assign(node.dataset, { rolloutId: String(spec.id), outcome: String(spec.outcome),
      sessions: String(spec.count), rank: String(rank), weight: String(weight) });
    const title = make('span', 'advantage-rollout-title', `Rollout ${spec.id}`);
    const count = make('span', 'advantage-session-count', `N = ${spec.count}`);
    const sessionRow = make('div', 'advantage-session-blocks');
    const sessions = Array.from({ length: spec.count }, (_, sessionIndex) => {
      const block = make('span', 'advantage-session-block');
      sessionRow.append(block);
      return { block, born: 1000 + sessionIndex * paces[index] * paceUnit };
    });
    const reward = make('span', 'advantage-reward', `R = ${spec.outcome}`);
    const header = make('div', 'advantage-card-header');
    const stats = make('span', 'advantage-card-stats');
    stats.append(count, reward);
    header.append(title, stats);
    const value = make('span', 'advantage-value');
    const number = make('strong', '', signed(base));
    value.append(make('small', '', 'Advantage'), number);
    node.append(header, sessionRow, value);
    scene.append(node);
    return { ...spec, index, node, header, stats, countNode: count, sessionRow, sessions, reward, value, number, base, final,
      sampleIndex: sampleOrder.findIndex(item => item.id === spec.id),
      sortedIndex: [...group].sort((a, b) => a.count - b.count || a.id - b.id).findIndex(item => item.id === spec.id) };
  });

  let elapsed = 0;
  let cycle = 0;
  let phase = '';
  let width = 0;
  let frame = null;
  let previousTime = null;
  let inView = false;

  function render(time) {
    let stageIndex = stages.length - 1;
    while (stages[stageIndex][0] > time) stageIndex -= 1;
    const [stageStart, stagePhase, text] = stages[stageIndex];
    if (phase !== stagePhase) {
      phase = stagePhase;
      root.dataset.phase = phase;
      message.textContent = text;
    }
    const next = stages[stageIndex + 1]?.[0] ?? duration;
    message.style.opacity = String(Math.min(stageIndex === 0 && cycle === 0 ? 1 : ease((time - stageStart) / 300), ease((next - time) / 300)));
    const fadeOut = 1 - ease((time - timing.reset) / 600);
    const separate = ease((time - timing.grouping) / 1000);
    const stack = ease((time - timing.grouping - 1000) / 1000);
    const entrance = ease(time / 500);
    const scored = ease((time - timing.scoring) / 600);
    const gap = width < 480 ? 14 : 20;
    const groupWidth = (width - gap) / 2;
    const gutter = 22;
    const groupedWidth = groupWidth - gutter;
    const sampleWidth = Math.min(width, 480);
    const sampleX = (width - sampleWidth) / 2;
    prompt.style.left = '0px';
    prompt.style.width = `${width}px`;
    prompt.style.opacity = String((1 - ease((time - timing.grouping) / 300)) * fadeOut);
    labels.forEach((node, index) => {
      node.style.top = '32px';
      node.style.left = `${index * (groupWidth + gap) + (index === 0 ? gutter : 0)}px`;
      node.style.width = `${groupedWidth}px`;
      node.style.opacity = String(ease((time - timing.grouped) / 500) * fadeOut);
    });
    arrows.forEach((node, index) => {
      node.style.left = `${index === 0 ? 0 : width - 18}px`;
      node.style.top = `${groupTop}px`;
      node.style.height = `${arrowHeight}px`;
      node.style.opacity = String(.28 * ease((time - timing.preference) / 600) * fadeOut);
    });

    cards.forEach(card => {
      const sampleY = 32 + card.sampleIndex * 60;
      const groupX = card.outcome ? gutter : groupWidth + gap;
      const groupY = groupTop + card.sortedIndex * groupStride;
      const cardWidth = mix(sampleWidth, groupedWidth, separate);
      const narrowInfo = ease((180 - cardWidth) / 20) * scored;
      const cardHeight = mix(52, groupCardHeight, stack) + 6 * narrowInfo * (1 - stack);
      // First separate outcomes horizontally, then gather each sorted column.
      // Row order is preserved in each column, so no card crosses another.
      const x = mix(sampleX, groupX, separate);
      const y = mix(sampleY, groupY, stack);
      card.node.style.left = `${x}px`;
      card.node.style.top = `${y}px`;
      card.node.style.width = `${cardWidth}px`;
      card.node.style.height = `${cardHeight}px`;
      card.node.style.opacity = String(entrance * fadeOut);
      card.node.style.setProperty('--score', String(scored));
      card.node.dataset.complete = String(time >= card.sessions[card.sessions.length - 1].born + sessionEntrance);
      card.countNode.style.opacity = String(ease((time - timing.counted) / 500));
      const rewardSpace = 36 * (1 - separate);
      const sessionGap = mix(4, 3, separate);
      const blockWidth = Math.min(mix(36, 10, separate),
        (cardWidth - 24 - rewardSpace - (maxSessions - 1) * sessionGap) / maxSessions);
      const blockHeight = Math.min(mix(12, 10, separate), blockWidth);
      card.sessionRow.style.gap = `${sessionGap}px`;
      card.sessionRow.style.height = `${mix(12, 10, separate)}px`;
      card.sessionRow.style.marginTop = `${mix(9, 5, stack) - 5 * narrowInfo * (1 - stack)}px`;
      card.sessions.forEach(({ block, born }) => {
        const shown = ease((time - born) / sessionEntrance);
        block.style.opacity = String(shown);
        block.style.display = time >= born ? '' : 'none';
        block.style.width = `${blockWidth}px`;
        block.style.height = `${blockHeight}px`;
        block.style.transform = `scale(${mix(.5, 1, shown)})`;
      });
      const rewardOpacity = scored * (1 - ease((time - timing.grouped) / 400));
      const rewardSlot = time < timing.grouped + 400 ? scored : 0;
      card.reward.style.opacity = String(rewardOpacity);
      card.reward.style.maxWidth = `${34 * rewardSlot}px`;
      card.reward.style.marginLeft = `${8 * rewardSlot}px`;
      card.header.dataset.stacked = String(cardWidth < 160 && rewardSlot > 0);
      // When the reward disappears on narrow cards, ease the metadata and
      // session row back into their compact positions instead of jumping.
      const release = time >= timing.grouped + 400 && cardWidth < 160
        ? 14 * (1 - ease((time - timing.grouped - 400) / 300)) : 0;
      card.stats.style.transform = `translateY(${release}px)`;
      card.sessionRow.style.transform = `translateY(${release}px)`;
      card.value.style.opacity = String(ease((time - timing.advantage) / 600));
      const advantage = mix(card.base, card.final, ease((time - timing.modulation) / 1200));
      card.number.textContent = signed(advantage);
      card.node.dataset.advantage = String(Number(advantage.toFixed(2)));
    });
  }

  function stop() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    previousTime = null;
  }
  function tick(now) {
    frame = null;
    // A busy frame can be timestamped before the request's performance.now().
    elapsed += Math.max(0, now - previousTime);
    previousTime = now;
    if (elapsed >= duration) {
      cycle += Math.floor(elapsed / duration);
      elapsed %= duration;
      root.dataset.cycle = String(cycle);
    }
    render(elapsed);
    frame = requestAnimationFrame(tick);
  }
  function synchronize() {
    if (motion.matches) {
      stop();
      root.dataset.state = 'static';
      root.dataset.phase = 'summary';
      message.style.removeProperty('opacity');
      message.textContent = 'Rank within each outcome group. Modulate the advantage without changing its sign.';
      return;
    }
    if (!inView || document.hidden) {
      stop();
      root.dataset.state = elapsed ? 'paused' : 'ready';
      return;
    }
    root.dataset.state = 'playing';
    if (frame === null) {
      width = scene.clientWidth;
      render(elapsed);
      previousTime = performance.now();
      frame = requestAnimationFrame(tick);
    }
  }
  root.dataset.cycle = '0';
  if (!motion.matches) {
    root.dataset.state = 'ready';
    width = scene.clientWidth;
    render(0);
  }
  const observer = new IntersectionObserver(entries => {
    inView = entries[0].isIntersecting;
    synchronize();
  }, { threshold: .15 });
  observer.observe(root);
  document.addEventListener('visibilitychange', synchronize);
  motion.addEventListener('change', () => {
    elapsed = 0;
    cycle = 0;
    phase = '';
    root.dataset.cycle = '0';
    synchronize();
  });
  const resize = () => {
    if (motion.matches) return;
    width = scene.clientWidth;
    render(elapsed);
  };
  if ('ResizeObserver' in window) new ResizeObserver(resize).observe(scene);
  else window.addEventListener('resize', resize);
})();
