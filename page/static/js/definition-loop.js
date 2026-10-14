/* One rollout can contain any number of sessions. The three visible examples
 * stand for Session 1, Session 2, …, Session n, not a three-session limit.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-definition-loop]');
  if (!root || !('IntersectionObserver' in window)) return;

  const scene = root.querySelector('.definition-live');
  const message = root.querySelector('.definition-loop-message');
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const svgNS = 'http://www.w3.org/2000/svg';
  const duration = 21000;
  const sessionHeight = 36;
  const rolloutTop = 16;
  const rolloutHeight = sessionHeight + 24;
  const sessionTop = rolloutTop + (rolloutHeight - sessionHeight) / 2;
  const births = [700, 3000, 5700];
  const stages = [
    [0, 'rollout', 'Inference: one rollout, multiple sessions.'],
    [8000, 'completed', 'The rollout completes.'],
    [11000, 'segments', 'Each session becomes one training segment.'],
    [15000, 'training', 'All segments contribute to the policy update.'],
    [18000, 'reset', 'A new rollout begins.'],
  ];
  const clamp = value => Math.max(0, Math.min(1, value));
  const ease = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
  const mix = (a, b, t) => a + (b - a) * t;
  const make = (tag, className, value) => {
    const element = document.createElement(tag);
    element.className = className;
    if (value !== undefined) element.textContent = value;
    return element;
  };

  const rollout = make('div', 'definition-rollout');
  rollout.style.top = `${rolloutTop}px`;
  rollout.style.height = `${rolloutHeight}px`;
  const rolloutLabel = make('span', 'definition-rollout-label', 'One rollout');
  rollout.append(rolloutLabel);
  const ellipsis = make('span', 'definition-ellipsis', '…');
  ellipsis.style.top = `${rolloutTop + rolloutHeight / 2 - 9}px`;
  const links = document.createElementNS(svgNS, 'svg');
  links.classList.add('definition-paths');
  const paths = births.map((_, index) => {
    const path = document.createElementNS(svgNS, 'path');
    path.dataset.contribution = String(index + 1);
    links.append(path);
    return path;
  });
  const trainer = make('div', 'definition-trainer');
  trainer.append(make('strong', '', 'RL Trainer'), make('small', '', 'Policy update'));
  trainer.dataset.updated = 'false';
  const units = ['1', '2', 'n'].map((index, position) => {
    const unit = make('div', 'definition-unit');
    unit.dataset.sessionIndex = index;
    unit.dataset.kind = 'session';
    const title = make('span', 'definition-unit-title', `Session ${index}`);
    const segment = make('span', 'definition-unit-segment-title', `Segment ${index}`);
    const segmentBars = make('span', 'definition-segment-bars');
    for (let i = 0; i < 5; i += 1) segmentBars.append(make('i', ''));
    const advantage = make('span', 'definition-advantage', `A${index}`);
    unit.append(title, segment, segmentBars, advantage);
    const packet = make('span', 'definition-packet');
    scene.append(packet);
    return { unit, title, segment, segmentBars, advantage, packet, path: paths[position] };
  });
  scene.prepend(rollout, links, ellipsis);
  scene.append(...units.map(item => item.unit), trainer);

  let elapsed = 0;
  let cycle = 0;
  let frame = null;
  let previousTime = null;
  let inView = false;
  let width = 0;
  let phase = '';

  function rowPositions(count, unitWidth, gap) {
    const center = width / 2;
    if (count === 1) return [center - unitWidth / 2];
    if (count === 2) return [center - unitWidth - gap / 2, center + gap / 2];
    const total = 3 * unitWidth + 2 * gap + 12;
    const left = (width - total) / 2;
    return [left, left + unitWidth + gap, left + 2 * unitWidth + 2 * gap + 12];
  }

  function packetPoint(x, progress) {
    const points = [[x, 127], [x, 168], [width / 2, 168], [width / 2, 195]];
    const distances = points.slice(1).map(([nextX, nextY], index) =>
      Math.hypot(nextX - points[index][0], nextY - points[index][1]));
    let remaining = distances.reduce((sum, value) => sum + value, 0) * clamp(progress);
    for (let i = 0; i < distances.length; i += 1) {
      if (remaining <= distances[i]) {
        const fraction = distances[i] ? remaining / distances[i] : 0;
        return [mix(points[i][0], points[i + 1][0], fraction), mix(points[i][1], points[i + 1][1], fraction)];
      }
      remaining -= distances[i];
    }
    return points.at(-1);
  }

  function render(time) {
    let stageIndex = stages.length - 1;
    while (stages[stageIndex][0] > time) stageIndex -= 1;
    const [stageStart, nextPhase, caption] = stages[stageIndex];
    if (phase !== nextPhase) {
      phase = nextPhase;
      root.dataset.phase = phase;
      message.textContent = caption;
    }
    const nextBoundary = stages[stageIndex + 1]?.[0] ?? duration;
    message.style.opacity = String(Math.min(
      stageIndex === 0 && cycle === 0 ? 1 : ease((time - stageStart) / 300),
      ease((nextBoundary - time) / 300),
    ));

    const gap = width < 400 ? 10 : 18;
    const unitWidth = Math.min(110, (width - 24 - 2 * gap - 12) / 3);
    const visibleCount = births.filter(born => time >= born).length;
    const activeCount = Math.max(1, visibleCount);
    const positions = rowPositions(activeCount, unitWidth, gap);
    const prior = rowPositions(Math.max(1, activeCount - 1), unitWidth, gap);
    const move = activeCount > 1 ? ease((time - births[activeCount - 1]) / 450) : 1;
    const cardX = units.map((_, index) => {
      const target = positions[index] ?? width / 2 - unitWidth / 2;
      if (activeCount <= 1 || index >= activeCount) return target;
      if (index === activeCount - 1) return mix(target + 10, target, move);
      return mix(prior[index], target, move);
    });
    const fadeOut = 1 - ease((time - 18000) / 700);
    const segmentProgress = index => ease((time - 11000 - index * 200) / 600);
    const finalLeft = cardX[0] - 11;
    const finalRight = cardX[Math.max(0, visibleCount - 1)] + unitWidth + 11;
    rollout.style.left = `${finalLeft}px`;
    rollout.style.width = `${Math.max(unitWidth + 22, finalRight - finalLeft)}px`;
    rollout.style.opacity = String((1 - ease((time - 11000) / 500)) * fadeOut);
    rolloutLabel.textContent = time >= 8000 ? 'One completed rollout' : 'One rollout';
    ellipsis.style.left = `${rowPositions(3, unitWidth, gap)[2] - gap - 12}px`;
    ellipsis.style.width = `${gap + 12}px`;
    ellipsis.style.opacity = String(ease((time - 5300) / 350) * (1 - ease((time - 11000) / 500)) * fadeOut);

    links.setAttribute('viewBox', `0 0 ${width} 260`);
    const pathOpacity = ease((time - 15000) / 500) * fadeOut;
    links.style.opacity = String(pathOpacity);
    trainer.style.left = `${(width - 140) / 2}px`;
    trainer.style.opacity = String(ease((time - 15000) / 500) * fadeOut);
    trainer.dataset.updated = String(time >= 17000 && time < 18000);

    units.forEach((item, index) => {
      const progress = segmentProgress(index);
      // The inference row reserves space for an ellipsis. Once it disappears,
      // center the middle segment on the trainer and space both sides equally.
      const segmentX = (width - unitWidth) / 2 + (index - 1) * (unitWidth + gap + 6);
      const x = mix(cardX[index], segmentX, progress);
      const y = mix(sessionTop, 72, progress);
      const height = mix(sessionHeight, 52, progress);
      item.unit.style.left = `${x}px`;
      item.unit.style.top = `${y}px`;
      item.unit.style.width = `${unitWidth}px`;
      item.unit.style.height = `${height}px`;
      item.unit.style.opacity = String(ease((time - births[index]) / 450) * fadeOut);
      item.unit.dataset.kind = progress > .5 ? 'segment' : 'session';
      item.title.style.opacity = String(1 - progress);
      item.segment.style.opacity = String(progress);
      item.segmentBars.style.opacity = String(progress);
      item.advantage.style.opacity = String(ease((time - 11800 - index * 200) / 450) * fadeOut);
      const centerX = x + unitWidth / 2;
      const d = `M${centerX} 127V168H${width / 2}V195`;
      if (item.path.getAttribute('d') !== d) item.path.setAttribute('d', d);
      item.path.dataset.delivered = String(time >= 16400 + index * 120 && time < 18000);
      const packetStart = 15200 + index * 120;
      const packetEnd = packetStart + 1200;
      const point = packetPoint(centerX, ease((time - packetStart) / 1200));
      item.packet.style.left = `${point[0] - 6}px`;
      item.packet.style.top = `${point[1] - 4}px`;
      item.packet.style.opacity = String(ease((time - packetStart) / 180) * (1 - ease((time - packetEnd) / 180)) * fadeOut);
    });
  }

  function stop() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    previousTime = null;
  }
  function tick(now) {
    frame = null;
    elapsed += now - previousTime;
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
      message.textContent = 'One rollout. Multiple sessions. One policy update.';
      message.style.removeProperty('opacity');
      return;
    }
    if (!inView || document.hidden) {
      stop();
      root.dataset.state = elapsed === 0 ? 'ready' : 'paused';
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
