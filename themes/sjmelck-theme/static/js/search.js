document.addEventListener("DOMContentLoaded", async () => {
  const page = document.querySelector(".search-page");
  const input = document.getElementById("search-query");
  const status = document.getElementById("search-status");
  const results = document.getElementById("search-results");
  if (!page || !input || !status || !results) return;

  const escapeHTML = (value) => {
    const element = document.createElement("div");
    element.textContent = value ?? "";
    return element.innerHTML;
  };

  const render = (items, query) => {
    results.replaceChildren();
    if (!query) {
      status.textContent = "Enter a term to search the blog.";
      return;
    }
    if (!items.length) {
      status.textContent = `No articles found for “${query}”.`;
      return;
    }

    status.textContent = `${items.length} ${items.length === 1 ? "article" : "articles"} found for “${query}”.`;
    results.innerHTML = items.map((item) => {
      const tags = (item.tags || []).map((tag) => `<span class="tag-chip">${escapeHTML(tag)}</span>`).join("");
      return `<article class="post-card"><div class="post-card__body">
        <h2 class="post-card__title"><a href="${escapeHTML(item.permalink)}">${escapeHTML(item.title)}</a></h2>
        <p class="post-card__description">${escapeHTML(item.description)}</p>
        <div class="post-meta"><span>${escapeHTML(item.author)}</span><span aria-hidden="true"> · </span><time datetime="${escapeHTML(item.date)}">${escapeHTML(item.date)}</time><span aria-hidden="true"> · </span><span>${item.readingTime} min read</span></div>
        ${tags ? `<div class="tag-list" aria-label="Tags">${tags}</div>` : ""}
      </div></article>`;
    }).join("");
  };

  try {
    const response = await fetch(page.dataset.indexUrl);
    if (!response.ok) throw new Error(`Search index returned ${response.status}`);
    const documents = await response.json();
    const fuse = new Fuse(documents, {
      keys: [
        { name: "title", weight: 0.4 },
        { name: "tags", weight: 0.22 },
        { name: "description", weight: 0.18 },
        { name: "author", weight: 0.1 },
        { name: "content", weight: 0.1 }
      ],
      threshold: 0.35,
      ignoreLocation: true
    });

    const search = () => {
      const query = input.value.trim();
      const items = query ? fuse.search(query, { limit: 24 }).map(({ item }) => item) : [];
      render(items, query);
      const url = new URL(window.location.href);
      query ? url.searchParams.set("q", query) : url.searchParams.delete("q");
      window.history.replaceState({}, "", url);
    };

    input.value = new URLSearchParams(window.location.search).get("q") || "";
    input.addEventListener("input", search);
    search();
  } catch (error) {
    status.textContent = "Search is temporarily unavailable.";
    console.error(error);
  }
});
