/* Inference-only demonstrations: one rollout, three independently accumulated
 * contexts. Compaction replaces a prefix; forking returns results, not histories.
 * The first single-session demonstration stays independent and unchanged.
 */
(() => {
  'use strict';
  if (!('IntersectionObserver' in window)) return;

  const ease = value => {
    const t = Math.max(0, Math.min(1, value));
    return t * t * (3 - 2 * t);
  };
  const mix = (a, b, t) => a + (b - a) * t;
  const duration = 38000;
  const overviewTime = 27000;
  const fadeOutTime = 34000;
  const resetTime = 35000;
  const entrance = 450;
  const sceneHeight = 508;
  const laneStride = 154;
  const cardCenter = 76;
  const horizontalPadding = 16;
  const verticalPadding = 12;
  const restart = 'The demonstration starts again with a fresh system prompt.';
  const conclusion = 'One rollout branches into multiple sessions, each with its own interaction history.';
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const svgNS = 'http://www.w3.org/2000/svg';

  const token = (born, role, text, hint, kind, final = false) => ({ born, role, text, hint, kind, final });
  const sys = born => token(born, 'sys', 'SYS', 'System prompt');
  const action = (born, hint = 'Action', final = false) => token(born, 'action', 'a', hint, undefined, final);
  const observation = born => token(born, 'observation', 'o', 'Observation');
  const definitions = {
    compaction: {
      boundary: 7000,
      staticText: 'Compaction replaces the context, so one rollout spans multiple sessions.',
      stages: [
        [0, 'initial', 'With context compaction, the agent first accumulates one continuous history.'],
        [7000, 'boundary', 'The agent detects the context limit, then ends this session.'],
        [11000, 'compaction', 'A separate compaction session turns the previous history into a summary.'],
        [17000, 'resumed', 'The summary replaces the old prefix. A new session continues the same task.'],
        [overviewTime, 'overview', conclusion],
        [resetTime, 'reset', restart],
      ],
      lanes: [
        { id: 'execution', label: 'Session 1 · execution', begin: 0, close: 8500,
          tokens: [sys(0), action(1500), observation(2500), action(3500), observation(4500), action(5500), observation(6500)] },
        { id: 'compaction', label: 'Session 2 · compaction', begin: 11000, close: 17000,
          tokens: [sys(11000), token(11500, 'history', 'History', 'Quoted history', 'history'),
            token(14500, 'summary', 'Summary', 'Generated summary', 'summary')] },
        { id: 'resumed', label: 'Session 3 · resumed execution', begin: 17000, close: overviewTime,
          tokens: [sys(17000), token(18000, 'summary', 'Summary', 'New prefix', 'summary'),
            action(19500), observation(21000), action(22500), observation(24000), action(25500, 'Final answer', true)] },
      ],
    },
    forking: {
      boundary: 6000,
      staticText: 'Forking creates independent contexts, so one rollout spans multiple sessions.',
      stages: [
        [0, 'initial', 'With subagent forking, the main agent first accumulates one continuous history.'],
        [6000, 'fork', 'The main agent delegates two subtasks through a fork action.'],
        [10000, 'branches', 'Each subagent accumulates its own history in a separate context window.'],
        [18000, 'return', 'Only the subagents’ results return as an observation—not their full histories.'],
        [22000, 'resumed', 'The main agent continues the same session and produces the final answer.'],
        [overviewTime, 'overview', conclusion],
        [resetTime, 'reset', restart],
      ],
      lanes: [
        { id: 'main', label: 'Main agent · one continuous session', begin: 0, close: overviewTime,
          tokens: [sys(0), action(1500), observation(2500), action(3500), observation(4500),
            token(6000, 'fork', 'Fork', 'Delegate tasks', 'fork'),
            // Make the receiving observation visible before either result returns.
            token(17000, 'observation', 'o', 'Subagent results', 'results'),
            action(23000), observation(24000), action(25500, 'Final answer', true)] },
        { id: 'subagent-a', label: 'Subagent A · its own session', begin: 8000, close: 18000,
          tokens: [sys(8000), token(8450, 'task', 'Task A', 'Assigned task', 'task'),
            action(10000), observation(11600), action(13200), observation(14800), action(16000, 'Result A', true)] },
        { id: 'subagent-b', label: 'Subagent B · its own session', begin: 8500, close: 18500,
          tokens: [sys(8500), token(8950, 'task', 'Task B', 'Assigned task', 'task'),
            action(10500), observation(12000), action(13800), observation(15400), action(17000, 'Result B', true)] },
      ],
    },
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function initialize(figure) {
    const mode = figure.dataset.multiSession;
    const definition = definitions[mode];
    if (!definition) return;
    const scene = figure.querySelector('.multi-session-scene');
    const message = figure.querySelector('.session-loop-message');
    const messageStage = figure.querySelector('.session-loop-stage');
    const links = document.createElementNS(svgNS, 'svg');
    links.classList.add('multi-links');
    const defs = document.createElementNS(svgNS, 'defs');
    const marker = document.createElementNS(svgNS, 'marker');
    const markerId = `multi-arrow-${mode}`;
    for (const [key, value] of Object.entries({ id: markerId, markerWidth: 6, markerHeight: 6, refX: 5, refY: 3, orient: 'auto' })) marker.setAttribute(key, value);
    const tip = document.createElementNS(svgNS, 'path');
    tip.setAttribute('d', 'M0 0L6 3L0 6Z');
    marker.append(tip);
    defs.append(marker);
    links.append(defs);
    scene.append(links);
    const paths = new Map();
    function path(name, d, opacity, draw = 1) {
      if (!paths.has(name)) {
        const node = document.createElementNS(svgNS, 'path');
        node.dataset.link = name;
        node.setAttribute('marker-end', `url(#${markerId})`);
        node.setAttribute('pathLength', '1');
        links.append(node);
        paths.set(name, node);
      }
      const node = paths.get(name);
      if (node.getAttribute('d') !== d) node.setAttribute('d', d);
      node.style.opacity = String(opacity);
      node.style.strokeDasharray = '1';
      node.style.strokeDashoffset = String(1 - draw);
    }

    const lanes = definition.lanes.map((spec, index) => {
      const root = element('div', 'multi-lane');
      root.dataset.sessionId = spec.id;
      const label = element('p', 'multi-lane-label', spec.label);
      const window = element('div', 'multi-lane-window');
      root.append(label, window);
      const cards = spec.tokens.map(specToken => {
        const chip = element('div', 'session-chip');
        chip.dataset.role = specToken.role;
        if (specToken.kind) chip.dataset.kind = specToken.kind;
        if (specToken.final) chip.setAttribute('data-final', '');
        const text = element('span', 'session-chip-label', specToken.text);
        const hint = element('span', 'session-chip-hint', specToken.hint);
        chip.append(text, hint);
        root.append(chip);
        return { ...specToken, chip, text, hint, x: 0, y: 0, width: 0 };
      });
      scene.append(root);
      return { ...spec, root, labelNode: label, window, cards, index, top: 0 };
    });
    const limit = element('div', 'multi-limit', 'Context limit detected');
    const budget = element('div', 'multi-context-budget');
    const budgetFill = element('span', 'multi-budget-fill');
    budget.append(budgetFill);
    if (mode === 'compaction') {
      messageStage.append(limit);
      lanes[0].window.append(budget);
    }
    const transfers = new Map();
    function transfer(name, text, role, source, target, fraction, waypoints = [], curveX = null) {
      const result = role === 'observation';
      const compact = result || role === 'summary';
      if (!transfers.has(name)) {
        const node = element('div', 'multi-transfer', text);
        if (result) node.classList.add('multi-result-transfer');
        if (role === 'summary') node.classList.add('multi-summary-transfer');
        node.dataset.role = role;
        scene.append(node);
        transfers.set(name, node);
      }
      const node = transfers.get(name);
      const size = compact ? 14 : Math.min(70, source.width + 12);
      const p = ease(fraction);
      let point = target;
      if (curveX !== null) {
        const q = 1 - p;
        point = {
          x: q ** 3 * source.x + 3 * q * p * curveX + p ** 3 * target.x,
          y: q ** 3 * source.y + 3 * q ** 2 * p * source.y + 3 * q * p ** 2 * target.y + p ** 3 * target.y,
        };
      } else {
        const route = [source, ...waypoints, target];
        const distances = route.slice(1).map((point, index) => Math.hypot(point.x - route[index].x, point.y - route[index].y));
        let remaining = distances.reduce((sum, distance) => sum + distance, 0) * p;
        for (let index = 0; index < distances.length; index += 1) {
          if (remaining <= distances[index]) {
            const local = distances[index] ? remaining / distances[index] : 0;
            point = { x: mix(route[index].x, route[index + 1].x, local), y: mix(route[index].y, route[index + 1].y, local) };
            break;
          }
          remaining -= distances[index];
        }
      }
      node.style.width = `${size}px`;
      node.style.height = compact ? '10px' : '30px';
      node.style.left = `${point.x - size / 2}px`;
      node.style.top = `${point.y - (compact ? 5 : 15)}px`;
      node.style.opacity = String(ease(fraction / .15) * (1 - ease((fraction - .8) / .2)));
    }

    let elapsed = 0;
    let cycle = 0;
    let frame = null;
    let previousTime = null;
    let inView = false;
    let phase = '';
    let width = scene.clientWidth;
    let linkWidth = -1;
    let messageX = 0;
    let messageY = 0;

    function measureMessage() {
      messageX = (messageStage.clientWidth - message.offsetWidth) / 2;
      messageY = figure.clientHeight / 2 - messageStage.offsetTop - message.offsetHeight / 2;
    }

    function renderLane(lane, time, focus, detailOpacity) {
      const first = lane.index === 0;
      const shift = first ? ease((time - definition.boundary) / 1200) : 1;
      const normalTop = first ? mix(184, 0, shift) : lane.index * laneStride;
      // Keep the middle history fixed while the outer two gather around it.
      const compactTop = laneStride + (lane.index - 1) * 84;
      lane.top = mix(normalTop, compactTop, focus);
      const laneWidth = width - 44;
      lane.root.style.left = '22px';
      lane.root.style.width = `${laneWidth}px`;
      lane.root.style.top = `${lane.top}px`;
      const archived = lane.close === undefined ? 0 : ease((time - lane.close) / 700);
      lane.root.dataset.status = archived > .5 ? 'closed' : 'active';
      const labelText = `${lane.label}${archived > .5 ? ' (Completed)' : ''}`;
      if (lane.labelNode.textContent !== labelText) lane.labelNode.textContent = labelText;
      lane.labelNode.style.opacity = String(detailOpacity);
      lane.root.style.opacity = String(ease((time - lane.begin) / entrance) * mix(1 - .35 * archived, 1, focus));
      const present = lane.cards.filter(card => card.born <= time);
      const count = present.length;
      const packed = ease((time - lane.begin - 3000) / 3500);
      const base = mix(64, 32, packed);
      const gap = mix(12, 4, packed);
      const naturalWidths = lane.cards.map(card => card.kind === 'summary' ? 76 : card.kind === 'history' ? 68
        : card.kind === 'task' ? 66 : card.kind === 'fork' ? 56 : card.final ? 48 : base);
      const total = present.reduce((sum, _, index) => sum + naturalWidths[index], 0) + gap * Math.max(0, count - 1);
      const zoom = Math.min(1, (laneWidth - 2 * horizontalPadding) / Math.max(1, total));
      const latest = present[count - 1];
      const enter = latest ? ease((time - latest.born) / entrance) : 0;
      const latestWidth = count ? naturalWidths[count - 1] + (count > 1 ? gap : 0) : 0;
      const visibleTotal = total * zoom - latestWidth * zoom * (1 - enter);
      let x = (laneWidth - visibleTotal) / 2;
      const start = x;
      const height = mix(44, 30, packed);
      lane.cards.forEach((card, index) => {
        const shown = index < count;
        const grow = index === count - 1 ? enter : 1;
        const cardWidth = naturalWidths[index] * zoom;
        card.x = 22 + x + cardWidth * (shown ? grow : 1) / 2;
        card.y = lane.top + cardCenter;
        card.width = cardWidth;
        card.height = height;
        card.chip.style.left = `${x}px`;
        card.chip.style.top = `${cardCenter - height / 2}px`;
        card.chip.style.width = `${cardWidth * (shown ? grow : 0)}px`;
        card.chip.style.height = `${height}px`;
        card.chip.style.opacity = String(shown ? grow : 0);
        card.text.style.opacity = String(cardWidth < 25 && !card.kind && !card.final ? 0 : 1);
        card.hint.style.opacity = String(shown && index === count - 1 ? (1 - archived) * detailOpacity : 0);
        const hintCenter = Math.max(55, Math.min(laneWidth - 55, x + cardWidth / 2));
        card.hint.style.left = `${hintCenter - x}px`;
        if (shown) {
          const nextGap = index === count - 2 ? gap * enter : gap;
          x += cardWidth + nextGap * zoom;
        }
      });
      lane.window.style.left = `${start - horizontalPadding}px`;
      lane.window.style.width = `${visibleTotal + 2 * horizontalPadding}px`;
      lane.window.style.top = `${cardCenter - height / 2 - verticalPadding}px`;
      lane.window.style.height = `${height + 2 * verticalPadding}px`;
      lane.frameLeft = 22 + start - horizontalPadding;
      lane.frameRight = lane.frameLeft + visibleTotal + 2 * horizontalPadding;
      lane.frameTop = lane.top + cardCenter - height / 2 - verticalPadding;
    }

    function render(time) {
      let stageIndex = definition.stages.length - 1;
      while (definition.stages[stageIndex][0] > time) stageIndex -= 1;
      const stage = definition.stages[stageIndex];
      if (phase !== stage[1]) {
        phase = stage[1];
        figure.dataset.phase = phase;
        message.textContent = stage[2];
        message.style.textAlign = time >= resetTime ? 'center' : 'left';
        measureMessage();
      }
      const next = definition.stages[stageIndex + 1]?.[0] ?? duration;
      const fadeIn = stageIndex === 0 && cycle === 0 ? 1 : ease((time - stage[0]) / (time >= resetTime ? 600 : 300));
      const sceneOpacity = 1 - ease((time - fadeOutTime) / 600);
      message.style.opacity = String(Math.min(fadeIn, ease((next - time) / 300)) * (time < resetTime ? sceneOpacity : 1));
      const centered = time >= resetTime ? 1 - ease((time - (duration - 1000)) / 1000) : 0;
      // Read layout only on a text or viewport change, never for every frame.
      message.style.transform = `translate(${messageX * centered}px, ${messageY * centered}px)`;
      scene.style.opacity = String(sceneOpacity);
      const focus = ease((time - overviewTime) / 1200);
      const detailOpacity = 1 - ease((time - overviewTime) / 600);
      scene.style.setProperty('--session-focus', String(focus));
      scene.dataset.conclusion = String(time >= overviewTime);
      links.style.opacity = String(detailOpacity);
      budget.style.opacity = String(detailOpacity);
      if (linkWidth !== width) {
        links.setAttribute('viewBox', `0 0 ${width} ${sceneHeight}`);
        linkWidth = width;
      }
      lanes.forEach(lane => renderLane(lane, time, focus, detailOpacity));

      if (mode === 'compaction') {
        limit.style.opacity = String(ease((time - 7000) / 400) * (1 - ease((time - 11000) / 400)));
        limit.textContent = lanes[0].root.dataset.status === 'closed' ? 'Context limit · session ended' : 'Context limit detected';
        lanes[0].window.dataset.contextLimit = String(time >= 7000 && time < 11000);
        const used = lanes[0].cards.reduce((sum, card) => sum + ease((time - card.born) / entrance), 0) / lanes[0].cards.length;
        budgetFill.style.width = `${used * 100}%`;
        // Brief handoffs join visible context boundaries. They do not target
        // future cards or leave permanent wiring around the completed history.
        const bend = (source, target) => Math.min(width - 8, Math.max(source.x, target.x) + 24);
        const handoffOpacity = (start, end) => ease((time - start) / 400) * (1 - ease((time - end) / 400));
        const historySource = { x: lanes[0].frameRight, y: lanes[0].top + cardCenter };
        const historyTarget = { x: lanes[1].frameRight, y: lanes[1].top + cardCenter };
        const historyBend = bend(historySource, historyTarget);
        path('history', `M${historySource.x} ${historySource.y}C${historyBend} ${historySource.y} ${historyBend} ${historyTarget.y} ${historyTarget.x} ${historyTarget.y}`, handoffOpacity(11450, 13200));
        const summary = lanes[1].cards[2];
        const source = { x: summary.x + summary.width / 2, y: summary.y, width: 14 };
        const target = { x: lanes[2].frameRight, y: lanes[2].top + cardCenter };
        const summaryBend = bend(source, target);
        path('summary', `M${source.x} ${source.y}C${summaryBend} ${source.y} ${summaryBend} ${target.y} ${target.x} ${target.y}`, handoffOpacity(17450, 18800));
        transfer('summary', 'Summary', 'summary', source, target, (time - 17450) / 1000, [], summaryBend);
      } else {
        const fork = lanes[0].cards[5];
        const result = lanes[0].cards[6];
        const resultTarget = { x: result.x, y: result.y - result.height / 2 };
        // Both branches share the same outer trunks. Horizontal turns happen
        // above the session frames, not in the padding around their cards.
        const leftGutter = 10;
        const rightGutter = width - 10;
        const forkRouteY = lanes[0].frameTop - 8;
        for (let index = 1; index <= 2; index += 1) {
          const child = lanes[index];
          const task = child.cards[1];
          const answer = child.cards[child.cards.length - 1];
          const startReturn = index === 1 ? 18000 : 18500;
          const childRouteY = child.frameTop - 8;
          path(`fork-${index}`, `M${fork.x} ${fork.y - fork.height / 2}V${forkRouteY}H${leftGutter}V${childRouteY}H${task.x}V${task.y - task.height / 2}`,
            ease((time - child.begin) / 650), ease((time - child.begin) / 650));
          path(`return-${index}`, `M${answer.x + answer.width / 2 + 4} ${answer.y}H${rightGutter}V${forkRouteY}H${resultTarget.x}V${resultTarget.y}`,
            ease((time - startReturn) / 650), ease((time - startReturn) / 1000));
          transfer(`result-${index}`, `Result ${index === 1 ? 'A' : 'B'}`, 'observation',
            { x: answer.x + answer.width / 2 + 4, y: answer.y, width: 14 },
            resultTarget, (time - startReturn) / 1600,
            [{ x: rightGutter, y: answer.y }, { x: rightGutter, y: forkRouteY }, { x: resultTarget.x, y: forkRouteY }]);
        }
      }
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
        figure.dataset.cycle = String(cycle);
      }
      render(elapsed);
      frame = requestAnimationFrame(tick);
    }
    function synchronize() {
      if (motion.matches) {
        stop();
        figure.dataset.state = 'static';
        figure.dataset.phase = 'summary';
        message.style.removeProperty('transform');
        message.style.removeProperty('opacity');
        message.style.removeProperty('text-align');
        message.textContent = definition.staticText;
        return;
      }
      if (!inView || document.hidden) {
        stop();
        figure.dataset.state = elapsed === 0 ? 'ready' : 'paused';
        return;
      }
      figure.dataset.state = 'playing';
      if (frame === null) {
        width = scene.clientWidth;
        render(elapsed);
        previousTime = performance.now();
        frame = requestAnimationFrame(tick);
      }
    }
    const observer = new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      synchronize();
    }, { threshold: .15 });
    figure.dataset.cycle = '0';
    if (!motion.matches) {
      figure.dataset.state = 'ready';
      width = scene.clientWidth;
      render(0);
    }
    observer.observe(figure);
    document.addEventListener('visibilitychange', synchronize);
    motion.addEventListener('change', () => {
      elapsed = 0;
      cycle = 0;
      phase = '';
      figure.dataset.cycle = '0';
      synchronize();
    });
    const resize = () => {
      if (motion.matches) return;
      width = scene.clientWidth;
      measureMessage();
      render(elapsed);
    };
    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(scene);
    else window.addEventListener('resize', resize);
  }

  document.querySelectorAll('[data-multi-session]').forEach(initialize);
})();
