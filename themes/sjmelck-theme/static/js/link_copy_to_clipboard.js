document.addEventListener("click", async (event) => {
  const anchor = event.target.closest("[data-copy-heading]");
  if (!anchor) return;

  event.preventDefault();
  const url = new URL(anchor.getAttribute("href"), window.location.href).href;
  try {
    await navigator.clipboard.writeText(url);
    const toast = document.getElementById("link-copy-toast");
    if (toast && window.bootstrap) {
      bootstrap.Toast.getOrCreateInstance(toast, { delay: 2200 }).show();
    }
  } catch {
    window.location.hash = anchor.hash;
  }
});
