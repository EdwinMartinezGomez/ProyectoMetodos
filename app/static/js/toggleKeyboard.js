// toggleKeyboard.js
document.addEventListener("DOMContentLoaded", () => {
    const toggleBtn = document.getElementById("toggle-keyboard-btn");
    const keyboardContainer = document.getElementById("keyboard-container");
    const mainContainer = document.querySelector(".main-container");

    // Inicialmente ocultar el teclado
    keyboardContainer.style.display = "none";

    toggleBtn.addEventListener("click", () => {
        const isKeyboardVisible = keyboardContainer.style.display === "flex" || keyboardContainer.style.display === "block";

        if (isKeyboardVisible) {
            // Ocultar el teclado
            keyboardContainer.style.display = "none";
            // Remover las clases que ajustan el ancho y la disposición
            mainContainer.classList.remove("keyboard-visible");
            keyboardContainer.classList.remove("expanded");
            // Actualizar el estado del botón
            toggleBtn.setAttribute("aria-pressed", "false");
            toggleBtn.setAttribute("aria-label", "Mostrar Teclado Virtual");
        } else {
            // Mostrar el teclado
            keyboardContainer.style.display = "flex"; // Cambiar a flex para mejor disposición
            // Agregar las clases que ajustan el ancho y la disposición
            mainContainer.classList.add("keyboard-visible");
            keyboardContainer.classList.add("expanded"); // Añadir clase para expansión centrada
            // Actualizar el estado del botón
            toggleBtn.setAttribute("aria-pressed", "true");
            toggleBtn.setAttribute("aria-label", "Ocultar Teclado Virtual");
        }
    });
});
