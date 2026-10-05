const appRoot = document.getElementById('app');
if (!appRoot) {
  throw new Error('Missing #app element');
}

// Override with e.g. `window.PYTHAGORAS_API_BASE = 'http://localhost:8000'`
// before this script loads (see README).
const API_BASE = String(window.PYTHAGORAS_API_BASE || 'http://127.0.0.1:8000').replace(/\/+$/, '');

appRoot.innerHTML = `
  <h1>Project Pythagoras</h1>
  <p>Starter frontend shell for the musicology and distribution agent.</p>
  <button id="status-button">Check API</button>
  <pre id="status-output">Waiting...</pre>
`;

const button = document.getElementById('status-button');
const output = document.getElementById('status-output');

button.addEventListener('click', async () => {
  output.textContent = 'Checking...';
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}`);
    }
    const data = await response.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    output.textContent = `Request failed: ${error.message}`;
  }
});
