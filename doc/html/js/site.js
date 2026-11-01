const menu = document.querySelector('.menu');
const navigation = document.querySelector('#navigation');
menu?.addEventListener('click', () => {
  const expanded = menu.getAttribute('aria-expanded') === 'true';
  menu.setAttribute('aria-expanded', String(!expanded));
  navigation.classList.toggle('open', !expanded);
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menu?.getAttribute('aria-expanded') === 'true') {
    menu.setAttribute('aria-expanded', 'false');
    navigation.classList.remove('open');
    menu.focus();
  }
});
for (const button of document.querySelectorAll('.copy')) {
  button.addEventListener('click', async () => {
    const content = button.closest('.code-wrap').querySelector('code').textContent;
    try {
      await navigator.clipboard.writeText(content);
      button.textContent = 'Copied';
      document.querySelector('#announcement').textContent = 'Code copied to clipboard.';
      setTimeout(() => { button.textContent = 'Copy'; }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(button.closest('.code-wrap').querySelector('code'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Selected';
      document.querySelector('#announcement').textContent = 'Code selected. Use your copy shortcut.';
    }
  });
}

const pythonKeywords = new Set('and as assert async await break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield'.split(' '));
const cppKeywords = new Set('struct class template typename const constexpr int float double void bool auto return if else for while break continue static unsigned long true false __global__ __device__ __host__ __forceinline__ __shared__'.split(' '));
const pythonBuiltins = new Set('bool dict enumerate float int len list max min print range set str super tuple ValueError RuntimeError'.split(' '));
for (const element of document.querySelectorAll('pre code')) {
  const language = element.closest('.code-wrap').querySelector('.code-label').textContent.trim().toLowerCase();
  const source = element.textContent;
  const tokens = /("""[\s\S]*?"""|'''[\s\S]*?'''|(?:[frb]|rf|fr)?"(?:\\.|[^"\\])*"|(?:[frb]|rf|fr)?'(?:\\.|[^'\\])*'|#[^\n]*|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b|\b[A-Za-z_][A-Za-z_0-9]*\b|--[\w-]+|\$[A-Za-z_][A-Za-z_0-9]*)/g;
  const fragment = document.createDocumentFragment();
  let end = 0;
  for (const match of source.matchAll(tokens)) {
    fragment.append(document.createTextNode(source.slice(end, match.index)));
    const value = match[0];
    let kind = '';
    if (value.startsWith('#')) kind = language === 'cuda c++' ? 'keyword' : 'comment';
    else if (/^(?:[frb]|rf|fr)?["']/.test(value)) kind = 'string';
    else if (/^\d/.test(value)) kind = 'number';
    else if (language === 'cuda c++' && cppKeywords.has(value)) kind = 'keyword';
    else if (language === 'python' && pythonKeywords.has(value)) kind = 'keyword';
    else if (language === 'python' && pythonBuiltins.has(value)) kind = 'builtin';
    else if (language === 'yaml' && /^\s*:/.test(source.slice(match.index + value.length))) kind = 'keyword';
    else if (language === 'shell' && (/^(python|make|torchrun)$/.test(value) || value.startsWith('--'))) kind = 'keyword';
    else if (source.slice(match.index + value.length).startsWith('(')) kind = 'function';
    if (kind) {
      const span = document.createElement('span');
      span.className = 'syntax-' + kind;
      span.textContent = value;
      fragment.append(span);
    } else fragment.append(document.createTextNode(value));
    end = match.index + value.length;
  }
  fragment.append(document.createTextNode(source.slice(end)));
  element.replaceChildren(fragment);
}
