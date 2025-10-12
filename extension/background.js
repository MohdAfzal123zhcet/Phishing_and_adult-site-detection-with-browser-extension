// extension/background.js
chrome.webNavigation.onCompleted.addListener(async (details) => {
  if (!details.url || !details.url.startsWith("http")) return;
  try {
    const resp = await fetch(`http://localhost:8000/score?url=${encodeURIComponent(details.url)}`);
    if (!resp.ok) return;
    const json = await resp.json();
    console.log("securebrowse verdict", json);
    if (json.verdict === "block") {
      if (json.category === "adult" || json.url.match(/(porn|xxx|sex|adult)/i)) {
        chrome.scripting.executeScript({
          target: { tabId: details.tabId },
          func: () => {
            document.documentElement.innerHTML = `<div style="padding:40px;font-family:Arial"><h1>Blocked: Adult content</h1><p>This site is blocked by SecureBrowse.</p></div>`;
          }
        });
        return;
      }
      // phishing block/warn UI
      chrome.scripting.executeScript({
        target: { tabId: details.tabId },
        func: (payload) => {
          const p = JSON.parse(payload);
          const html = `<div style="position:fixed;inset:0;z-index:2147483647;background:white;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px;">
            <h1>Warning — Suspicious Page</h1>
            <p>Reason: ${p.category}</p>
            <p>Score: ${p.score.toFixed(2)}</p>
            <p>Model: ${p.used_model}</p>
            <div style="margin-top:12px"><button id="sb-proceed">Proceed anyway</button></div>
          </div>`;
          document.documentElement.innerHTML = html;
          document.getElementById('sb-proceed').onclick = () => { location.reload(); };
        },
        args: [JSON.stringify(json)]
      });
    } else if (json.verdict === "warn") {
      chrome.notifications.create('', {type:"basic", title:"SecureBrowse warning", message:`Site suspicious (${json.score.toFixed(2)})`});
    }
  } catch (e) { console.error(e); }
});
