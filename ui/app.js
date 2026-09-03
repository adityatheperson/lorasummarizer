const source = document.querySelector("#source-text");
const count = document.querySelector("#character-count");
const button = document.querySelector("#compare-button");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const baseSummary = document.querySelector("#base-summary");
const loraSummary = document.querySelector("#lora-summary");
const baseTime = document.querySelector("#base-time");
const loraTime = document.querySelector("#lora-time");

source.addEventListener("input", () => {
  count.textContent = `${source.value.length.toLocaleString()} / 20,000`;
});

button.addEventListener("click", async () => {
  const text = source.value.trim();
  error.hidden = true;
  if (!text) {
    error.textContent = "Enter some English text to summarize.";
    error.hidden = false;
    source.focus();
    return;
  }

  button.disabled = true;
  status.textContent = "Comparing both models…";
  baseSummary.textContent = "Generating…";
  loraSummary.textContent = "Waiting for the base model…";
  baseSummary.classList.add("empty");
  loraSummary.classList.add("empty");

  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to compare summaries.");

    baseSummary.textContent = payload.base.summary;
    loraSummary.textContent = payload.lora.summary;
    baseTime.textContent = `${payload.base.seconds.toFixed(2)}s`;
    loraTime.textContent = `${payload.lora.seconds.toFixed(2)}s`;
    baseSummary.classList.remove("empty");
    loraSummary.classList.remove("empty");
    status.textContent = "Comparison complete.";
  } catch (problem) {
    error.textContent = problem.message;
    error.hidden = false;
    status.textContent = "Comparison failed.";
    baseSummary.textContent = "No result.";
    loraSummary.textContent = "No result.";
  } finally {
    button.disabled = false;
  }
});
