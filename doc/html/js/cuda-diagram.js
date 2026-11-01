(() => {
  const host = document.querySelector('#cuda-scan-diagram');
  if (!host) return;
  const workerSelect = host.querySelector('[data-workers]');
  const chunkSelect = host.querySelector('[data-chunk-length]');
  const slider = host.querySelector('[data-scan-step]');
  const canvas = host.querySelector('[data-scan-canvas]');
  const description = host.querySelector('[data-scan-description]');
  const phaseButtons = [...host.querySelectorAll('[data-phase]')];
  const play = host.querySelector('[data-play]');
  const previous = host.querySelector('[data-previous]');
  const next = host.querySelector('[data-next]');
  const ns = 'http://www.w3.org/2000/svg';
  const colors = ['#3567cc', '#a56317', '#27805d', '#8154bd', '#b24762', '#237d91', '#6267a5', '#68722c'];
  let timer = null;
  let steps = [];
  let current = 0;
  if (host.clientWidth < 520) workerSelect.value = '2';
  function element(tag, attributes, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes || {})) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function stop() {
    clearInterval(timer);
    timer = null;
    play.textContent = 'Play from start';
    play.setAttribute('aria-pressed', 'false');
  }
  function buildSteps(workers) {
    const count = Math.log2(workers);
    const list = [
      {kind: 'tokens', name: 'Input tokens', text: 'Token representations are partitioned into contiguous chunks. Each displayed worker owns two chunks.'},
      {kind: 'summary', name: '(i) Chunk summaries', text: 'All chunks compute their affine pairs (Φ, S) independently. S is the chunk contribution from zero entry state; Φ transports an arbitrary entry state.'},
      {kind: 'fold', name: '(ii) Worker pairs', text: 'Each worker composes its two consecutive chunk pairs in chronological order.'}
    ];
    for (let round = 1; round <= count; round++) list.push({kind: 'scan', round, name: `Scan step ${round}`, text: `Each worker combines its running pair with the preceding pair ${2 ** (round - 1)} worker position${round === 1 ? '' : 's'} to the left. All reads use the previous scan step; token order is preserved.`});
    list.push(
      {kind: 'prefix', name: 'Preceding pairs', text: 'Shift the inclusive scan by one worker to obtain the exclusive prefix. The first worker receives the identity pair (I, 0).'},
      {kind: 'entry', name: 'Chunk entry states', text: 'Apply each worker’s preceding pair to V₀, then apply its first chunk map to obtain the second chunk’s entry state.'},
      {kind: 'output', name: '(iii) Chunk replay', text: 'All chunks replay concurrently from their recovered entry states. The ladder readout produces every yₜ. This completes the illustrative forward schedule.'}
    );
    return list;
  }
  function draw() {
    const workers = Number(workerSelect.value);
    const length = Number(chunkSelect.value);
    const chunks = 2 * workers;
    const width = Math.max(canvas.clientWidth, 108 + chunks * 64);
    const left = 100;
    const pitch = (width - left - 12) / chunks;
    const rowHeight = 84;
    const top = 56;
    const height = top + steps.length * rowHeight + 24;
    const svg = element('svg', {viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-labelledby': 'cuda-scan-title cuda-scan-desc'});
    svg.append(element('title', {id: 'cuda-scan-title'}, 'CUDA affine chunk scan: summarize, compose and replay'));
    svg.append(element('desc', {id: 'cuda-scan-desc'}, `${workers} worker threads, ${chunks} chunks, ${length} tokens per chunk. Solid arrows show local flow; dashed arrows show an exchange between workers. Read from top to bottom.`));
    const defs = element('defs');
    const marker = element('marker', {id: 'scan-arrow', markerWidth: 6, markerHeight: 6, refX: 5, refY: 3, orient: 'auto'});
    marker.append(element('path', {d: 'M0,0 L6,3 L0,6 Z', fill: '#6a7d94'}));
    defs.append(marker);
    svg.append(defs);
    const layers = steps.map((_, i) => {
      const group = element('g', {opacity: i <= current ? 1 : .15, 'data-scan-row': i});
      svg.append(group);
      return group;
    });
    const cx = chunk => left + (chunk + .5) * pitch;
    const wx = worker => (cx(2 * worker) + cx(2 * worker + 1)) / 2;
    const y = row => top + row * rowHeight;
    function line(group, x1, y1, x2, y2, exchange = false) {
      group.append(element('path', {d: `M${x1},${y1 + 23} L${x2},${y2 - 23}`, fill: 'none', stroke: '#6a7d94', 'stroke-width': exchange ? 1.2 : .9, 'stroke-dasharray': exchange ? '4 3' : 'none', 'marker-end': 'url(#scan-arrow)'}));
    }
    function text(group, x, yy, value, attributes = {}) {
      group.append(element('text', {x, y: yy, 'text-anchor': 'middle', ...attributes}, value));
    }
    function box(group, x, yy, worker, label, pair = null, boxWidth = pitch - 8) {
      group.append(element('rect', {x: x - boxWidth / 2, y: yy - 22, width: boxWidth, height: 44, rx: 4, fill: '#fff', stroke: colors[worker], 'stroke-width': 1.3}));
      if (pair) {
        for (const [symbol, dy] of [['Φ', -4], ['S', 13]]) {
          const labelNode = element('text', {x, y: yy + dy, 'text-anchor': 'middle', fill: '#23374e'});
          labelNode.append(element('tspan', {}, symbol));
          labelNode.append(element('tspan', {'baseline-shift': 'sub', 'font-size': 9}, `${pair[0]}→${pair[1]}`));
          group.append(labelNode);
        }
      } else text(group, x, yy + 4, label);
    }
    for (let worker = 0; worker < workers; worker++) text(svg, wx(worker), 20, `Thread ${worker + 1}`, {fill: colors[worker], 'font-weight': 650});
    steps.forEach((step, row) => {
      const group = layers[row];
      const yy = y(row);
      if (row === current) group.append(element('rect', {x: 0, y: yy - 33, width, height: 66, rx: 4, fill: '#eaf0ff'}));
      const label = step.name.split(' ');
      const midpoint = Math.ceil(label.length / 2);
      text(group, 44, yy - 2, label.slice(0, midpoint).join(' '), {'font-size': 11, fill: '#53657a'});
      text(group, 44, yy + 13, label.slice(midpoint).join(' '), {'font-size': 11, fill: '#53657a'});
      if (step.kind === 'tokens' || step.kind === 'summary' || step.kind === 'entry' || step.kind === 'output') {
        for (let chunk = 0; chunk < chunks; chunk++) {
          const worker = Math.floor(chunk / 2);
          const from = chunk * length;
          const to = from + length;
          if (row > 0) {
            if (step.kind === 'entry') {
              if (chunk % 2 === 0) line(group, wx(worker), y(row - 1), cx(chunk), yy);
              else group.append(element('path', {d: `M${cx(chunk - 1) + (pitch - 8) / 2},${yy} L${cx(chunk) - (pitch - 8) / 2},${yy}`, stroke: '#6a7d94', 'marker-end': 'url(#scan-arrow)'}));
            } else line(group, cx(chunk), y(row - 1), cx(chunk), yy);
          }
          if (step.kind === 'summary') box(group, cx(chunk), yy, worker, '', [from, to]);
          else if (step.kind === 'entry') box(group, cx(chunk), yy, worker, `V${from}`);
          else box(group, cx(chunk), yy, worker, `${step.kind === 'tokens' ? 'x' : 'y'}${from + 1}…${to}`);
        }
      } else {
        for (let worker = 0; worker < workers; worker++) {
          if (step.kind === 'fold') {
            line(group, cx(2 * worker), y(row - 1), wx(worker), yy);
            line(group, cx(2 * worker + 1), y(row - 1), wx(worker), yy);
          } else if (step.kind === 'scan') {
            line(group, wx(worker), y(row - 1), wx(worker), yy);
            const distance = 2 ** (step.round - 1);
            if (worker >= distance) line(group, wx(worker - distance), y(row - 1), wx(worker), yy, true);
          } else if (worker > 0) line(group, wx(worker - 1), y(row - 1), wx(worker), yy, true);
          let from = 2 * worker * length;
          let to = 2 * (worker + 1) * length;
          if (step.kind === 'scan') from = Math.max(0, worker - 2 ** step.round + 1) * 2 * length;
          if (step.kind === 'prefix') { from = 0; to = 2 * worker * length; }
          box(group, wx(worker), yy, worker, worker === 0 && step.kind === 'prefix' ? '(I, 0)' : '', worker === 0 && step.kind === 'prefix' ? null : [from, to], Math.min(2 * pitch - 18, 126));
        }
      }
    });
    canvas.replaceChildren(svg);
    host.querySelector('[data-scan-size]').textContent = `${workers} workers · ${chunks} chunks · L = ${length} · T = ${chunks * length}`;
    slider.max = String(steps.length - 1);
    slider.value = String(current);
    host.querySelector('[data-step-label]').textContent = steps[current].name;
    description.textContent = steps[current].text;
    previous.disabled = current === 0;
    next.disabled = current === steps.length - 1;
    for (const button of phaseButtons) {
      const phase = button.dataset.phase;
      const active = phase === 'summary' ? current === 1 : phase === 'compose' ? current >= 2 && current < steps.length - 1 : current === steps.length - 1;
      button.setAttribute('aria-pressed', String(active));
    }
  }
  function rebuild() {
    stop();
    steps = buildSteps(Number(workerSelect.value));
    current = steps.length - 1;
    draw();
  }
  workerSelect.addEventListener('change', rebuild);
  chunkSelect.addEventListener('change', rebuild);
  slider.addEventListener('input', () => { stop(); current = Number(slider.value); draw(); });
  previous.addEventListener('click', () => { stop(); current = Math.max(0, current - 1); draw(); });
  next.addEventListener('click', () => { stop(); current = Math.min(steps.length - 1, current + 1); draw(); });
  for (const button of phaseButtons) button.addEventListener('click', () => {
    stop();
    current = button.dataset.phase === 'summary' ? 1 : button.dataset.phase === 'compose' ? steps.length - 2 : steps.length - 1;
    draw();
  });
  play.addEventListener('click', () => {
    if (timer) { stop(); return; }
    current = 0;
    draw();
    play.textContent = 'Pause';
    play.setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      current++;
      draw();
      if (current === steps.length - 1) stop();
    }, 1300);
  });
  new ResizeObserver(() => draw()).observe(canvas);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  rebuild();
})();
