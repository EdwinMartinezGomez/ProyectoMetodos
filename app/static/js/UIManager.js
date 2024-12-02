class UIManager {
    constructor() {
        this.elements = this.initializeElements();
        this.formHandler = new FormHandler(document.getElementById('form'));
        this.keyboard = new Keyboard();
        this.toggleKeyboard = new ToggleKeyboard(this.keyboard);
    }
     // Mostrar u ocultar teclado
     handleKeyboardToggle() {
        this.toggleKeyboard.toggle();
    }

    // Validar y enviar formulario
    handleFormSubmit() {
        this.formHandler.validateAndSubmit();
    }

    initializeElements(ids) {
        ids.forEach(id => {
            const element = document.getElementById(id);
            if (!element) throw new Error(`Elemento con ID ${id} no encontrado`);
            this.elements[id] = element;
        });
    }
    

    getElement(id) {
        return this.elements[id];
    }
}