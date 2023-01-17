
document.addEventListener('scroll', function() {

    const a = document.documentElement.scrollTop;
    const b = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const c = a / b * 100;
    const stringC = "" + c.toFixed(0) + "%";
//
//    console.log("Here")
//    console.log(stringC)

//    const progress_bar = document.getElementById("progressElement");
//    progress_bar.style.setAttribute('width', c);
//    console.log(progress_bar.style.width)

    document.getElementById("progressElement").style.width = stringC;
//    const progress_bar = document.getElementById("progressElement");
//    console.log(progress_bar.style)
})
