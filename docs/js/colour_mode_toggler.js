const dm_off = "bi-moon";
const dm_on = "bi-sun";

// Function to change the icon randomly
function changeIcon(img_toggle){

    if (img_toggle.classList.contains(dm_off)){
        img_toggle.classList.replace(dm_off, dm_on);
    }
    else if (img_toggle.classList.contains(dm_on)){
        img_toggle.classList.replace(dm_on, dm_off);
    }
    else{
        console.log("Something strange happened.")
    }
}

// Function to check whether the icon is correct for the colour mode selected.
function checkIcon(img_toggle){

    if (document.documentElement.getAttribute('data-bs-theme') == 'dark'){
        if (img_toggle.classList.contains(dm_off)){
            img_toggle.classList.replace(dm_off, dm_on);
        }
    }
    else{
        if (img_toggle.classList.contains(dm_on)){
            img_toggle.classList.replace(dm_on, dm_off);
        }
    }
}

// Function that returns the stored theme from the localStorage page or
const getPreferredTheme = function(storedTheme) {
if (storedTheme) {
  return storedTheme
}
return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

// Function that sets the colour theme in document.documentElement (it edits the data-bs-theme attribute)
const setTheme = function (theme) {
if (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
  document.documentElement.setAttribute('data-bs-theme', 'light')
} else {
  document.documentElement.setAttribute('data-bs-theme', theme)
}
}

// page-loadup checker
document.addEventListener('DOMContentLoaded', function() {

    const img_toggle = document.getElementById("colorSwitchToggle");

    if (window.matchMedia){

        // Check the stored color scheme of the document
        const storedTheme = localStorage.getItem('theme');

        // Check to see if the browser has a prefers-color-scheme option
        if (window.matchMedia('(prefers-color-scheme)').matches){

            // Set the theme
            setTheme(getPreferredTheme(storedTheme));
        }

        checkIcon(img_toggle);
    }
    else{
        console.log("No match-Queries are available")
    }

    // Monitor the picture
    // Add in a function to change the color mode in the document.
    img_toggle.onclick = function (){

        const userTheme = document.documentElement.getAttribute('data-bs-theme');

        if (userTheme == 'light'){
            document.documentElement.setAttribute('data-bs-theme', 'dark')
            localStorage.setItem('theme', 'dark')
        }
        else if (userTheme == 'dark'){
            document.documentElement.setAttribute('data-bs-theme', 'light')
            localStorage.setItem('theme', 'light')
        }

        console.log(getPreferredTheme(userTheme))

        checkIcon(img_toggle);
    }
})

//User setting change detector
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {

    // Check the stored color scheme of the document
    const storedTheme = localStorage.getItem('theme');

    if (storedTheme !== 'light' || storedTheme !== 'dark') {
      setTheme(getPreferredTheme())
    }
  })
