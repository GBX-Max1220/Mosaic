/* A single-session rollout becomes one loss-masked RL training segment.
 * Coordinates change for visual zooming only; the history is never rewritten.
 * Static HTML remains the baseline for reduced motion or unavailable scripting.
 */
(() => {
  'use strict';

  const figure = document.querySelector('[data-session-loop]');
  if (!figure || !('IntersectionObserver' in window)) return;
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scene = figure.querySelector('.session-loop-scene');
  const sequence = figure.querySelector('.session-token-sequence');
  const prefix = figure.querySelector('.session-prefix');
  const context = figure.querySelector('.session-context');
  const segmentLabel = figure.querySelector('.session-training-label');
  const maskHeading = figure.querySelector('.session-mask-heading');
  const trainer = figure.querySelector('.session-trainer');
  const arrow = figure.querySelector('.session-trainer-arrow');
  const message = figure.querySelector('.session-loop-message');
  const messageStage = figure.querySelector('.session-loop-stage');
  // Explanation boundaries and visual milestones share one active-playback clock.
  const timeline = {
    interaction: 3000,
    context: 8000,
    zoom: 9500,
    zoomDuration: 7200,
    result: 17000,
    session: 18000,
    training: 21000,
    morphDuration: 2000,
    mask: 23500,
    trainer: 26500,
    send: 29000,
    sendDuration: 1800,
    update: 31000,
    reset: 35000,
    duration: 38000,
  };
  const duration = timeline.duration;
  const messageFadeHalf = 300;
  const messageCenterDuration = 600;
  const messageReturnDuration = 1000;
  const cardEntrance = 450;
  const prefixHold = 200;
  const prefixExpansion = 250;
  const births = [0, timeline.interaction, 4500, 6000, timeline.context, 9500, 10250, 11000, 11750, 12500, 13250, 14000, 14750, 15500, 16250, timeline.result];
  const stages = [
    [0, 'system', 'The system prompt starts a new session.'],
    [timeline.interaction, 'interaction', 'An action is generated from the current prefix; its observation is appended next.'],
    [timeline.context, 'context', 'Each new interaction extends the same prefix, in the same context window.'],
    [timeline.result, 'result', 'The session ends with the final action returning the task result, completing one continuous history.'],
    [timeline.training, 'training', 'This completed session becomes one loss-masked training segment, which is then passed to the trainer.'],
    [timeline.update, 'update', 'One rollout. One session. One training segment.'],
    [timeline.reset, 'reset', 'The demonstration starts again with a fresh system prompt.'],
  ];

  const tokens = births.map((born, index) => {
    const final = index === births.length - 1;
    const role = index === 0 ? 'sys' : index % 2 === 1 ? 'action' : 'observation';
    const chip = document.createElement('div');
    chip.className = 'session-chip';
    chip.dataset.role = role;
    if (final) chip.setAttribute('data-final', '');
    const label = document.createElement('span');
    label.className = 'session-chip-label';
    label.textContent = role === 'sys' ? 'SYS' : role === 'action' ? 'a' : 'o';
    const hint = document.createElement('span');
    hint.className = 'session-chip-hint';
    hint.textContent = final ? 'Final answer' : role === 'sys' ? 'System prompt' : role === 'action' ? 'Action' : 'Observation';
    chip.append(label, hint);
    const mask = document.createElement('span');
    mask.className = 'session-mask';
    mask.dataset.role = role;
    mask.textContent = role === 'action' ? '1' : '0';
    sequence.append(chip, mask);
    return { born, role, final, chip, label, hint, mask };
  });

  const clamp = value => Math.max(0, Math.min(1, value));
  const ease = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
  const mix = (a, b, fraction) => a + (b - a) * fraction;
  let elapsed = 0;
  let cycle = 0;
  let inView = false;
  let frame = null;
  let previousTime = null;
  let width = 0;
  let phase = '';

  function render(time) {
    const stageIndex = stages.findLastIndex(item => time >= item[0]);
    const stage = stages[stageIndex];
    if (phase !== stage[1]) {
      phase = stage[1];
      figure.dataset.phase = phase;
      message.textContent = stage[2];
    }
    // Fade around each boundary using the active clock, including the loop seam.
    const nextStageTime = stages[stageIndex + 1]?.[0] ?? duration;
    const messageIn = stageIndex === 0 && cycle === 0 ? 1 : ease((time - stage[0]) / messageFadeHalf);
    const messageOut = ease((nextStageTime - time) / messageFadeHalf);
    message.style.opacity = String(Math.min(messageIn, messageOut));
    const centered = ease((time - timeline.update) / messageCenterDuration)
      * (1 - ease((time - (duration - messageReturnDuration)) / messageReturnDuration));
    message.style.textAlign = time >= timeline.update ? 'center' : 'left';
    const centerX = (messageStage.clientWidth - message.offsetWidth) / 2;
    const centerY = figure.clientHeight / 2 - messageStage.offsetTop - message.offsetHeight / 2;
    message.style.transform = `translate(${centerX * centered}px, ${centerY * centered}px)`;
    const abstract = ease((time - timeline.zoom) / timeline.zoomDuration);
    const morph = ease((time - timeline.training) / timeline.morphDuration);
    const send = ease((time - timeline.send) / timeline.sendDuration);
    const fade = 1 - ease((time - timeline.reset) / (duration - timeline.reset));
    const count = tokens.filter(token => token.born <= time).length;
    const current = tokens[count - 1];
    const enter = ease((time - current.born) / cardEntrance);
    const baseWidth = mix(64, 24, abstract);
    const gap = mix(12, 5, abstract);
    const naturalWidths = tokens.slice(0, count).map(token => token.final ? mix(84, 24, morph) : baseWidth);
    const naturalTotal = naturalWidths.reduce((sum, value) => sum + value, 0) + gap * (count - 1);
    // Reserve a label gutter as the sequence becomes a training segment.
    const maskLabelGutter = 72;
    const trainingGutter = maskLabelGutter * morph;
    const sequenceWidth = width - trainingGutter;
    const zoom = Math.min(1, sequenceWidth / naturalTotal);
    const fullWidths = naturalWidths.map(value => value * zoom);
    const visibleWidth = naturalTotal * zoom - fullWidths[count - 1] * (1 - enter);
    let x = trainingGutter + (sequenceWidth - visibleWidth) / 2;
    const start = x;
    const masksOpacity = ease((time - timeline.mask) / 650) * (1 - send);
    let accumulatedWidth = 0;

    tokens.forEach((token, index) => {
      if (index >= count) {
        token.chip.style.opacity = '0';
        token.mask.style.opacity = '0';
        return;
      }
      const last = index === count - 1;
      const grow = last ? enter : 1;
      const chipWidth = fullWidths[index] * grow;
      // A generated card joins the prefix only after its entrance and a hold.
      // Keep every card's acceptance clock independent, including during zooming.
      const accepted = ease((time - token.born - cardEntrance - prefixHold) / prefixExpansion);
      accumulatedWidth += (fullWidths[index] + (index === 0 ? 0 : gap * zoom)) * accepted;
      const height = token.final ? mix(50, 34, morph) : mix(54, 34, abstract);
      token.chip.style.left = `${x}px`;
      token.chip.style.top = `${112 - height / 2 + (last ? 8 * (1 - enter) : 0)}px`;
      token.chip.style.width = `${chipWidth}px`;
      token.chip.style.height = `${height}px`;
      token.chip.style.borderRadius = `${mix(7, 3, Math.max(abstract, morph))}px`;
      token.chip.style.opacity = String(grow);
      token.label.style.opacity = String(token.final ? 1 - morph : 1 - ease(abstract / .65));
      token.hint.style.opacity = String(last ? (token.final ? 1 - morph : 1 - abstract) : 0);
      const hintCenter = Math.max(55, Math.min(width - 55, x + chipWidth / 2));
      token.hint.style.left = `${hintCenter - x}px`;
      token.mask.style.left = `${x}px`;
      token.mask.style.width = `${chipWidth}px`;
      token.mask.style.opacity = String(masksOpacity);
      x += fullWidths[index] + gap * zoom;
    });

    prefix.style.left = `${start}px`;
    prefix.style.width = `${accumulatedWidth}px`;
    const prefixVisible = count > 1 ? ease((time - births[1]) / 200) : 0;
    prefix.style.opacity = String(prefixVisible * (1 - morph) * fade);
    prefix.querySelector('span').textContent = time >= timeline.session ? 'One completed session' : 'Accumulated prefix';
    context.style.left = `${start}px`;
    context.style.width = `${visibleWidth}px`;
    context.style.opacity = String(ease((time - timeline.context) / 700) * (1 - morph) * fade);
    segmentLabel.style.opacity = String(morph * (1 - send) * fade);
    maskHeading.style.left = `${start - maskLabelGutter}px`;
    maskHeading.style.opacity = String(masksOpacity);
    const trainerOpacity = ease((time - timeline.trainer) / 600)
      * (1 - ease((time - timeline.update) / messageCenterDuration)) * fade;
    trainer.style.opacity = String(trainerOpacity);
    arrow.style.opacity = String(trainerOpacity * (1 - send));
    sequence.style.transform = `translateY(${112 * send}px) scale(${1 - .7 * send})`;
    sequence.style.opacity = String((1 - ease((send - .85) / .15)) * fade);
  }

  function tick(now) {
    frame = null;
    // A busy frame can be timestamped before the request's performance.now().
    elapsed += Math.max(0, now - previousTime);
    previousTime = now;
    if (elapsed >= duration) {
      cycle += Math.floor(elapsed / duration);
      elapsed %= duration;
      figure.dataset.cycle = String(cycle);
    }
    render(elapsed);
    frame = requestAnimationFrame(tick);
  }

  function stop() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    previousTime = null;
  }

  function synchronize() {
    if (motion.matches) {
      stop();
      figure.dataset.state = 'static';
      figure.dataset.phase = 'summary';
      message.style.removeProperty('opacity');
      message.style.removeProperty('transform');
      message.style.removeProperty('text-align');
      message.textContent = 'One continuous interaction history becomes one loss-masked training segment.';
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
    // Resume from a fresh rollout after switching back from the static view.
    elapsed = 0;
    cycle = 0;
    phase = '';
    figure.dataset.cycle = '0';
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
