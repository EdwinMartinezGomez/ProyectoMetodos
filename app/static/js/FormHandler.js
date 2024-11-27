class FormHandler {
    constructor(uiManager) {
        this.uiManager = uiManager;
    }

    handleSubmit() {
        try {
            const method = this.uiManager.getElement('method').value;
            console.log(`Método seleccionado: ${method}`);
            // Realiza validaciones adicionales y prepara los datos
        } catch (error) {
            console.error('Error al manejar el formulario:', error);
        }
    }

    handleMethodChange(method) {
        console.log(`Cambiando a método: ${method}`);
        // Oculta y muestra campos relevantes según el método seleccionado
    }
}