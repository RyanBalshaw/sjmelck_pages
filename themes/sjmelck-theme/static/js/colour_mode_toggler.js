(() => {
  const storageKey = "theme";
  const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
  const toggle = document.getElementById("colorSwitchToggle");

  const getStoredTheme = () => localStorage.getItem(storageKey);

  const getPreferredTheme = () => {
    const storedTheme = getStoredTheme();
    if (storedTheme === "light" || storedTheme === "dark") {
      return storedTheme;
    }
    return mediaQuery.matches ? "dark" : "light";
  };

  const updateChromaTheme = (theme) => {
    const lightStyles = document.getElementById("chroma-light");
    const darkStyles = document.getElementById("chroma-dark");
    if (lightStyles) lightStyles.disabled = theme === "dark";
    if (darkStyles) darkStyles.disabled = theme !== "dark";
  };

  const updateToggle = (theme) => {
    if (!toggle) return;

    const icon = toggle.querySelector("i");
    const dark = theme === "dark";
    icon.className = dark ? "bi bi-sun" : "bi bi-moon-stars";
    toggle.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  };

  const setTheme = (theme) => {
    const resolvedTheme = theme === "auto"
      ? (mediaQuery.matches ? "dark" : "light")
      : theme;
    document.documentElement.setAttribute("data-bs-theme", resolvedTheme);
    updateChromaTheme(resolvedTheme);
    updateToggle(resolvedTheme);
  };

  setTheme(getPreferredTheme());

  toggle?.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-bs-theme");
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    localStorage.setItem(storageKey, nextTheme);
    setTheme(nextTheme);
  });

  mediaQuery.addEventListener("change", () => {
    const storedTheme = getStoredTheme();
    if (storedTheme !== "light" && storedTheme !== "dark") {
      setTheme("auto");
    }
  });
})();
