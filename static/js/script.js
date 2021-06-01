
//selecting all required elements
const dropArea = document.querySelector(".drag-area"),
dragText = dropArea.querySelector("header"),
button = dropArea.querySelector("button"),
input = dropArea.querySelector("input");

const lines = document.querySelector(".basura2");
const selector = document.querySelector(".basura");

button.onclick = ()=>{
  input.click(); //if user click on the button then the input also clicked
}

//If user Drag File Over DropArea
lines.addEventListener("dragover", (event)=>{
  event.preventDefault(); //preventing from default behaviour
  lines.classList.add("active2");
  selector.classList.add("grande");
  dragText.textContent = "Release to Upload Folder";
});

//If user leave dragged File from DropArea
lines.addEventListener("dragleave", ()=>{
    var rect = lines.getBoundingClientRect();
    event.preventDefault(); //preventing from default behaviour
    if (event.pageX < rect.right &&
        event.pageX > rect.left &&
        event.pageY < rect.bottom &&
        event.pageY > rect.top) {
            return false;
        }
  lines.classList.remove("active2");
  selector.classList.remove("grande");
  dragText.textContent = "Drag & Drop to Upload Folder";
});
