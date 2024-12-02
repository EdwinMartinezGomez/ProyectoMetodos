class EventManager {
    constructor(uiManager, mathFieldManager, formHandler,elements,formElement) {
        this.elements = elements;
        this.uiManager = uiManager;
        this.mathFieldManager = mathFieldManager;
        this.formHandler = formHandler;
        this.mathQuillManager = mathQuillManager;
        this.form = document.querySelector('form');
        this.form = formElement;
        this.formHandler = new FormHandler();
    }
    constructor() {
        this.formHandler = new FormHandler();
        this.calculationService = new CalculationService();
        this.resultRender = new ResultRender();
    }

    bindEvents() {
        this.formHandler.elements.form.addEventListener('submit', this.handleFormSubmit.bind(this));
    }

    async handleFormSubmit(event) {
        event.preventDefault();
        this.formHandler.clearErrors();
        this.formHandler.showLoading();

        try {
            const formData = await this.formHandler.validateAndPrepareFormData();
            const response = await this.calculationService.sendCalculationRequest(formData);
            await this.resultRender.handleCalculationResponse(response, formData.method);
        } catch (error) {
            this.formHandler.showError(error);
        } finally {
            this.formHandler.hideLoading();
        }
    }
    
    attachEventListeners() {
        this.form.addEventListener('submit', (event) => {
            event.preventDefault();
            // Obtener datos del formulario y procesarlos
            const method = document.querySelector('#methodSelect').value;
            const formData = this.prepareFormData();
            this.formHandler.processFormData(method, formData);
        });
    }

    prepareFormData() {
        // Preparar los datos del formulario
        const formData = {
            iterations: parseInt(document.getElementById('iterations').value),
            intervalStart: parseFloat(document.getElementById('intervalStart').value),
            intervalEnd: parseFloat(document.getElementById('intervalEnd').value),
            // Otros campos de formulario
        };
        return formData;
    }
    attachEventListeners() {
        this.form.addEventListener('submit', (event) => {
            event.preventDefault();
            // Llamada a la lógica de validación y cálculo
        });

        // Otros eventos
    }
     // Registrar un evento de clic
     registerClickEvent(element, callback) {
        element.addEventListener('click', callback);
    }

    // Registrar evento de cambio en un input
    registerInputChangeEvent(inputElement, callback) {
        inputElement.addEventListener('input', callback);
    }
    setupEventListeners() {
        try {
            this.elements.form.addEventListener('submit', this.handleFormSubmit.bind(this));
            this.elements.methodSelect.addEventListener('change', this.handleMethodChange.bind(this));
            this.elements.findIntervalBtn.addEventListener('click', this.handleFindInterval.bind(this));

            // Añadir listeners para los nuevos botones
            this.elements.addEquationBtn.addEventListener('click', this.addEquationField.bind(this));
            this.addVariableListeners();

            // Validaciones en tiempo real para initial_guess_system
            this.setupInitialGuessSystemValidation();

            // Inicializar el estado correcto al cargar la página
            this.handleMethodChange({ target: this.elements.methodSelect });

            // Configurar el Event Listener para el Toggle del Teclado Virtual
            this.elements.toggleKeyboardBtn.addEventListener('click', this.toggleKeyboard.bind(this));

            // Manejar la eliminación de ecuaciones
            this.elements.equationsContainer.addEventListener('click', this.handleEquationRemoval.bind(this));

            console.log("Event listeners configurados correctamente.");
        } catch (error) {
            throw new Error(`Error configurando event listeners: ${error.message}`);
        }
    }

    // Listener para el cambio de método
    handleMethodChange(event) {
        // Implementar lógica para cambiar el método de cálculo según el valor seleccionado
        console.log("Método de cálculo cambiado:", event.target.value);
    }

    // Listener para el envío del formulario
    handleFormSubmit(event) {
        event.preventDefault();
        // Implementar lógica de envío de formulario
        console.log("Formulario enviado.");
    }

    // Añadir listeners para las variables
    addVariableListeners() {
        const numVariablesInput = this.elements.form.querySelector('#numVariables');
        if (numVariablesInput) {
            numVariablesInput.addEventListener('change', this.updateVariables.bind(this));
            numVariablesInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault(); // Evitar que el formulario se envíe si está dentro de uno
                    this.updateVariables();
                }
            });
        }
    }

    // Validación en tiempo real para el campo initial_guess_system
    setupInitialGuessSystemValidation() {
        const initialGuessSystemInput = this.elements.initialGuessSystem.querySelector('input');
        if (initialGuessSystemInput) {
            initialGuessSystemInput.addEventListener('input', this.validateInitialGuessSystem.bind(this));
        }
    }

    // Validación de la entrada del sistema inicial
    validateInitialGuessSystem(event) {
        const input = event.target;
        const pattern = /^-?\d+(\.\d+)?(?:\s*,\s*-?\d+(\.\d+)?)*$/;
        if (!pattern.test(input.value)) {
            input.setCustomValidity('Ingrese valores numéricos separados por comas, por ejemplo: 1,1');
        } else {
            input.setCustomValidity('');
        }
    }

    // Método para actualizar las variables dinámicamente
    updateVariables() {
        // Implementar lógica para actualizar las variables del sistema
        console.log("Variables actualizadas.");
    }

    // Toggle para mostrar/ocultar el teclado virtual
    toggleKeyboard() {
        const isVisible = this.elements.keyboardContainer.classList.contains('show');
        this.elements.keyboardContainer.classList.toggle('show');
        this.elements.toggleKeyboardBtn.setAttribute('aria-pressed', !isVisible);
    }

    // Manejo de la eliminación de ecuaciones dinámicas
    handleEquationRemoval(event) {
        if (event.target.classList.contains('removeEquationBtn') || event.target.closest('.removeEquationBtn')) {
            const equationInputDiv = event.target.closest('.equation-input');
            if (equationInputDiv) {
                equationInputDiv.remove();
                this.updateEquationLabels();
                this.updateVariables(); // Actualizar variables después de eliminar una ecuación
            }
        }
    }

    // Actualizar las etiquetas de las ecuaciones
    updateEquationLabels() {
        const equations = this.elements.equationsContainer.querySelectorAll('.equation-input');
        equations.forEach((eq, index) => {
            const label = eq.querySelector('.input-group-text');
            label.textContent = `Ecuación ${index + 1}:`;
            const hiddenInput = eq.querySelector('input[type="hidden"]');
            hiddenInput.id = `equation_${index + 1}`;
        });
    }
    setupEventListeners() {
        this.elements.form.addEventListener('submit', this.handleFormSubmit.bind(this));
        this.elements.methodSelect.addEventListener('change', this.handleMethodChange.bind(this));
        this.elements.findIntervalBtn.addEventListener('click', this.handleFindInterval.bind(this));
        this.elements.addEquationBtn.addEventListener('click', this.addEquationField.bind(this));

        const numVariablesInput = this.elements.form.querySelector('#numVariables');
        if (numVariablesInput) {
            numVariablesInput.addEventListener('change', this.updateVariables.bind(this));
            numVariablesInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.updateVariables();
                }
            });
        }
    }

    handleFormSubmit(event) {
        event.preventDefault();
        // Lógica para manejar el envío del formulario
    }

    handleMethodChange(event) {
        // Lógica para manejar el cambio de método
    }

    setupEvents() {
        const form = this.uiManager.getElement('calculator-form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.formHandler.handleSubmit();
        });

        const methodSelect = this.uiManager.getElement('method');
        methodSelect.addEventListener('change', (e) => {
            this.formHandler.handleMethodChange(e.target.value);
        });
    }
}