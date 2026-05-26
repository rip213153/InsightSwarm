async function collectPage() {
  const selectedText = String(window.getSelection ? window.getSelection() : "");
  const visibleText = document.body ? document.body.innerText : "";
  const htmlExcerpt = document.documentElement ? document.documentElement.outerHTML.slice(0, 8000) : "";
  return {
    source_url: location.href,
    title: document.title || "",
    captured_at: new Date().toISOString(),
    visible_text: visibleText.slice(0, 12000),
    selected_text: selectedText.slice(0, 4000),
    html_excerpt: htmlExcerpt,
    collector: {
      name: "insightswarm_mv3_extension",
      version: "0.1.0"
    },
    page_metadata: {
      lang: document.documentElement ? document.documentElement.lang : "",
      referrer_origin: document.referrer ? new URL(document.referrer).origin : ""
    }
  };
}

async function sendCurrentPage() {
  const status = document.getElementById("status");
  const port = document.getElementById("port").value || "8765";
  status.textContent = "Collecting page...";
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: collectPage
  });
  const response = await fetch(`http://127.0.0.1:${port}/collect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(result)
  });
  const payload = await response.json();
  status.textContent = JSON.stringify(payload, null, 2);
}

document.getElementById("send").addEventListener("click", () => {
  sendCurrentPage().catch((error) => {
    document.getElementById("status").textContent = String(error);
  });
});
