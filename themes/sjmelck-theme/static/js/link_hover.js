document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll('.link-hover').forEach(function(element) {
        element.addEventListener('mouseenter', function() {
            // Show icon
            this.querySelector('.hoverable-icon').style.display = 'inline-block'; // Or "block", depending on your layout
        });
        element.addEventListener('mouseleave', function() {
            // Hide icon
            this.querySelector('.hoverable-icon').style.display = 'none';
        });
    });
});
