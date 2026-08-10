const app = document.getElementById('app');
app.innerHTML = `
  <h1>Project Pythagoras</h1>
  <p>Starter frontend shell for the musicology and distribution agent.</p>
  <button id="status-button">Check API</button>
  <pre id="status-output">Waiting...</pre>
`;

const button = document.getElementById('status-button');
const output = document.getElementById('status-output');

button.addEventListener('click', async () => {
  try {
    const response = await fetch('http://127.0.0.1:8000/health');
    const data = await response.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    output.textContent = `Request failed: ${error.message}`;
  }
});
