class FormHandler {
    constructor(uiManager) {
        this.uiManager = uiManager;
        this.formElement = formElement;
        this.elements = elements;
    }
    constructor() {
        this.elements = {
            form: document.querySelector('#form'),
            input: document.querySelector('#input')
        };
    }

    async validateAndPrepareFormData() {
        // Asumimos que los campos de entrada se validan y se preparan para ser enviados
        const formData = {
            method: this.elements.method.value,
            equations: this.elements.equations.value,
            variables: this.elements.variables.value
        };
        // Realizar validaciones adicionales si es necesario
        return formData;
    }

    clearErrors() {
        const existingErrors = document.querySelectorAll('.alert-danger');
        existingErrors.forEach(error => error.remove());

        // Limpiar la sección de resultados y gráficos
        this.elements.resultTable.innerHTML = '';
        this.elements.plotHtmlContainer.innerHTML = '';
        this.elements.resultsDiv.style.display = 'none';
    }

    showLoading() {
        const loadingSpinner = document.createElement('div');
        loadingSpinner.id = 'loading-spinner';
        loadingSpinner.className = 'text-center mt-3';
        loadingSpinner.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Calculando...</span>
            </div>
            <p class="mt-2">Calculando resultados...</p>
        `;
        this.elements.form.appendChild(loadingSpinner);
    }

    hideLoading() {
        const spinner = document.getElementById('loading-spinner');
        if (spinner) {
            spinner.remove();
        }
    }

    static validateAndPrepareFormData() {
        const method = document.querySelector('#methodSelect').value;
        const iterations = parseInt(document.querySelector('#iterations').value);
        const intervalStart = parseFloat(document.querySelector('#intervalStart').value);
        const intervalEnd = parseFloat(document.querySelector('#intervalEnd').value);

        if (isNaN(intervalStart) || isNaN(intervalEnd)) {
            throw new Error('Los valores de los intervalos deben ser números válidos');
        }

        if (iterations <= 0) {
            throw new Error('El número de iteraciones debe ser mayor a 0');
        }

        return {
            method,
            iterations,
            intervalStart,
            intervalEnd
        };
    }

    static processFormData(method, formData) {
        try {
            const results = CalculationService.solveMethod(method, formData);
            // Enviar los resultados para renderizarlos
            new ResultRender().renderResults(results);
        } catch (error) {
            new DOMManager().showError(error.message);
        }
    }
    async validateAndPrepareFormData() {
        const method = this.elements.methodSelect.value;
        const iterations = parseInt(this.elements.form.iterations.value, 10);

        // Lógica de validación y preparación de los datos
    }

     // Validar y enviar el formulario
     validateAndSubmit() {
        if (this.validateForm()) {
            this.submitForm();
        }
    }

    validateForm() {
        // Lógica de validación de formulario
        return true;
    }

    submitForm() {
        // Lógica para enviar el formulario
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