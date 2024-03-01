function CopyLink(copyText) {

    // Copy the text inside the text field
    navigator.clipboard.writeText(copyText);

    // Optional: display a message to the user
    alert('Link copied to clipboard!');
}