document.addEventListener("click", async (event) => {
  const button = event.target.closest(".code-copy");
  if (!button) return;

  const code = button.closest(".code-block")?.querySelector("code");
  if (!code) return;

  const original = button.innerHTML;
  try {
    await navigator.clipboard.writeText(code.innerText);
    button.innerHTML = '<i class="bi bi-check2" aria-hidden="true"></i> Copied';
  } catch {
    button.textContent = "Copy failed";
  }

  window.setTimeout(() => {
    button.innerHTML = original;
  }, 1800);
});
