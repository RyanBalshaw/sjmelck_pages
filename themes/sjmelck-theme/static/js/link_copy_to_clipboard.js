// Function to copy the link to the clipboard
function CopyLink(copyText) {

    // Copy the text inside the text field
    navigator.clipboard.writeText(copyText);

    // Show the toast that fades away
    const toastBootstrap = bootstrap.Toast.getOrCreateInstance(toastLiveExample)
    toastBootstrap.show()
}

// Functions to trigger the bootstrap toast element
const toastTrigger = document.getElementById('liveToastBtn')
const toastLiveExample = document.getElementById('liveToast')

if (toastTrigger) {
  const toastBootstrap = bootstrap.Toast.getOrCreateInstance(toastLiveExample)
  toastTrigger.addEventListener('click', () => {
    toastBootstrap.show()
  })
}
