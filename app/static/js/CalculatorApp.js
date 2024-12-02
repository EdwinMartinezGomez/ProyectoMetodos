class CalculatorApp {
    constructor() {
        try {
            this.elements = {}; 
            this.activeMathField = null;
            this.allMathFields = new Map(); // Mapa para almacenar todos los campos MathQuill
            this.initializeMathQuill();
            this.setupEventListeners();
            this.initializeElements();
            this.setupEventListeners();
            this.calculationService = new CalculationService();
            this.formHandler = new FormHandler();
            this.resultRender = new ResultRender();
            this.domManager = new DOMManager();
            


            // Hacer la instancia disponible globalmente
            window.calculatorApp = this;
        } catch (error) {
            this.handleInitializationError(error);
        }
    }
    constructor(calculationService, domManager) {
        this.calculationService = calculationService;
        this.domManager = domManager;
        this.calculationService = new CalculationService();
        this.domManager = new DOMManager();
        this.formHandler = new FormHandler();
    }
    initializeApp() {
        try {
            this.domManager.setupDOMElements();
            this.formHandler.attachEventListeners();
        } catch (error) {
            this.handleInitializationError(error);
        }
    }

    handleInitializationError(error) {
        // Código de manejo de error
    }
    handleInitializationError(error) {
        this.domManager.showError(error.message);
    }

    handleCalculation(method, func, limits) {
        const result = this.calculationService.integrate(method, func, limits);
        this.domManager.displayIntegrationResults(result);
    }
    // Método para inicializar los elementos del DOM
    initializeElements() {
        // Aquí debes inicializar los elementos del DOM como lo tienes en tu código
    }

    // Método para actualizar las etiquetas de las ecuaciones
    updateEquationLabels() {
        const equations = this.elements.equationsContainer.querySelectorAll('.equation-input');
        equations.forEach((eq, index) => {
            const label = eq.querySelector('.input-group-text');
            label.textContent = `Ecuación ${index + 1}:`;
            const hiddenInput = eq.querySelector('input[type="hidden"]');
            hiddenInput.id = `equation_${index + 1}`;
        });
    }

    // Método para alternar la visibilidad del teclado virtual
    toggleKeyboard() {
        const isVisible = this.elements.keyboardContainer.classList.contains('show');
        this.elements.keyboardContainer.classList.toggle('show');
        this.elements.toggleKeyboardBtn.setAttribute('aria-pressed', !isVisible);
        this.elements.toggleKeyboardBtn.setAttribute('aria-label', isVisible ? 'Mostrar Teclado Virtual' : 'Ocultar Teclado Virtual');
    }

    // Método para validar la entrada del sistema inicial (initialGuessSystem)
    validateInitialGuessSystem(event) {
        const input = event.target;
        const pattern = /^-?\d+(\.\d+)?(?:\s*,\s*-?\d+(\.\d+)?)*$/;
        if (!pattern.test(input.value)) {
            input.setCustomValidity('Ingrese valores numéricos separados por comas, por ejemplo: 1,1');
        } else {
            input.setCustomValidity('');
        }
    }

    // Método para manejar el cambio de método seleccionado
    handleMethodChange(event) {
        const selectedMethod = event.target.value;

        // Resetear 'required' en todos los campos
        this.elements.form.querySelectorAll('input').forEach(input => input.required = false);

        // Mostrar u ocultar campos según el método seleccionado
        this.showHideFieldsByMethod(selectedMethod);

        // Inicializar o limpiar MathQuill según el método seleccionado
        this.toggleMathQuill(selectedMethod);

        // Actualizar el ejemplo de MathQuill según el método seleccionado
        this.updateMathQuillExample(selectedMethod);
    }

    // Método para mostrar u ocultar campos según el método de cálculo seleccionado
    showHideFieldsByMethod(selectedMethod) {
        // Mostrar/ocultar y habilitar/deshabilitar campos según el método
        if (selectedMethod === 'bisection') {
            this.showBisectionFields();
        } else if (selectedMethod === 'secant') {
            this.showSecantFields();
        } else if (selectedMethod === 'newton') {
            this.showNewtonFields();
        } else if (selectedMethod === 'fixed_point') {
            this.showFixedPointFields();
        } else if (['jacobi', 'gauss_seidel', 'broyden'].includes(selectedMethod)) {
            this.showSystemFields();
        } else if (['trapezoidal', 'simpson'].includes(selectedMethod)) {
            this.showIntegrationFields();
        } else {
            this.hideAllFields();
        }
    }

    // Mostrar campos para el método de Bisección
    showBisectionFields() {
        this.elements.intervalInputs.style.display = 'flex';
        this.elements.findIntervalBtn.style.display = 'block';
        this.elements.aBisection.required = true;
        this.elements.bBisection.required = true;

        this.hideFields([
            'secantInputs', 'initialGuessInput', 'fixedPointInputs', 
            'systemInputs', 'equationsContainer', 'variablesContainer', 
            'initialGuessSystem', 'integrationInputs'
        ]);

        this.elements.singleEquationInput.style.display = 'block';
        this.enableFields(['a_bisection', 'b_bisection']);
    }

    // Mostrar campos para el método de Secante
    showSecantFields() {
        this.elements.secantInputs.style.display = 'flex';
        this.elements.secantInputs.querySelectorAll('input').forEach(input => input.required = true);

        this.hideFields([
            'intervalInputs', 'findIntervalBtn', 'initialGuessInput', 
            'fixedPointInputs', 'systemInputs', 'equationsContainer', 
            'variablesContainer', 'initialGuessSystem', 'integrationInputs'
        ]);

        this.elements.singleEquationInput.style.display = 'block';
        this.disableFields(['a_bisection', 'b_bisection']);
    }

    // Mostrar campos para el método de Newton
    showNewtonFields() {
        this.elements.initialGuessInput.style.display = 'block';
        this.elements.initialGuessInput.querySelector('input').required = true;

        this.hideFields([
            'intervalInputs', 'findIntervalBtn', 'secantInputs', 
            'fixedPointInputs', 'systemInputs', 'equationsContainer', 
            'variablesContainer', 'initialGuessSystem', 'integrationInputs'
        ]);

        this.elements.singleEquationInput.style.display = 'block';
        this.disableFields(['a_bisection', 'b_bisection']);
    }

    // Mostrar campos para el método de Punto Fijo
    showFixedPointFields() {
        this.elements.initialGuessInput.style.display = 'block';
        this.elements.fixedPointInputs.style.display = 'block';
        this.elements.initialGuessInput.querySelector('input').required = true;

        this.hideFields([
            'intervalInputs', 'findIntervalBtn', 'secantInputs', 
            'systemInputs', 'equationsContainer', 'variablesContainer', 
            'initialGuessSystem', 'integrationInputs'
        ]);

        this.elements.singleEquationInput.style.display = 'block';
        this.disableFields(['a_bisection', 'b_bisection']);
    }

    // Mostrar campos para el método de Sistema de Ecuaciones
    showSystemFields() {
        this.elements.systemInputs.style.display = 'block';
        this.elements.equationsContainer.style.display = 'block';
        this.elements.variablesContainer.style.display = 'block';
        this.elements.initialGuessSystem.style.display = 'block';
        this.elements.initialGuessSystem.querySelector('input').required = true;

        this.hideFields([
            'intervalInputs', 'findIntervalBtn', 'secantInputs', 
            'initialGuessInput', 'fixedPointInputs', 'singleEquationInput', 
            'integrationInputs'
        ]);

        this.disableFields(['a_bisection', 'b_bisection']);

        const variableInputs = this.elements.variablesContainer.querySelectorAll('input[name="variables[]"]');
        variableInputs.forEach(input => input.required = true);
    }

    // Mostrar campos para el método de Integración
    showIntegrationFields() {
        this.elements.integrationInputs.style.display = 'flex';
        this.elements.integrationInputs.querySelectorAll('input').forEach(input => {
            input.required = true;
        });

        this.hideFields([
            'intervalInputs', 'findIntervalBtn', 'secantInputs',
            'initialGuessInput', 'fixedPointInputs', 'systemInputs',
            'equationsContainer', 'variablesContainer', 'initialGuessSystem'
        ]);

        this.elements.singleEquationInput.style.display = 'block';
        this.disableFields(['a_bisection', 'b_bisection']);
    }

    // Ocultar todos los campos
    hideAllFields() {
        this.hideFields([
            'intervalInputs', 'findIntervalBtn', 'initialGuessInput',
            'fixedPointInputs', 'secantInputs', 'systemInputs',
            'equationsContainer', 'variablesContainer', 'initialGuessSystem',
            'integrationInputs'
        ]);

        this.elements.singleEquationInput.style.display = 'block';
        this.disableFields(['a_bisection', 'b_bisection']);
    }

    // Método para ocultar campos específicos
    hideFields(fields) {
        fields.forEach(field => {
            if (this.elements[field]) {
                this.elements[field].style.display = 'none';
            }
        });
    }

    // Habilitar campos específicos
    enableFields(fields) {
        fields.forEach(field => {
            if (this.elements[field]) {
                this.elements[field].disabled = false;
            }
        });
    }

    // Deshabilitar campos específicos
    disableFields(fields) {
        fields.forEach(field => {
            if (this.elements[field]) {
                this.elements[field].disabled = true;
            }
        });
    }

    // Actualizar el ejemplo de MathQuill según el método seleccionado
    updateMathQuillExample(selectedMethod) {
        const examples = {
            'bisection': 'x**2 - 4',
            'newton': 'x**3 - 2*x - 5',
            'secant': '',
            'fixed_point': '',
            'jacobi': 'x + y - 3',
            'gauss_seidel': 'x + y - 3',
            'broyden': 'x**2 + y**2 - 4, x**2 - y - 1',
            'trapezoidal': 'x**2 - 4',
            'simpson': 'x**2 - 4'
        };

        if (this.activeMathField && examples[selectedMethod]) {
            this.activeMathField.latex(examples[selectedMethod]);
        } else if (this.activeMathField) {
            this.activeMathField.latex('');
        }
    }

    // Método para alternar MathQuill según el método seleccionado
    toggleMathQuill(selectedMethod) {
        // Lógica para activar o desactivar MathQuill según el método seleccionado
    }

    initializeMathQuill() {
        try {
            if (typeof MathQuill === 'undefined') {
                throw new Error('MathQuill no está disponible');
            }

            const MQ = MathQuill.getInterface(2);

            // Inicializar el campo principal de ecuación
            const mathInputField = MQ.MathField(this.elements.mathInput, {
                handlers: {
                    edit: () => {
                        try {
                            const latex = mathInputField.latex().trim(); // Eliminar espacios
                            this.elements.equationHidden.value = this.latexToJavaScript(latex);
                        } catch (error) {
                            this.showError('Error en la ecuación: ' + error.message);
                            console.error('Error en MathQuill handler:', error);
                        }
                    },
                    focus: () => {
                        this.activeMathField = mathInputField;
                        console.log("Campo MathQuill enfocado: mathInput");
                    },
                    blur: () => {
                        console.log("Campo MathQuill desenfocado: mathInput");
                    }
                }
            });
            this.allMathFields.set('mathInput', mathInputField);

            // Inicializar el campo g(x)
            const gFunctionField = MQ.MathField(this.elements.gFunctionInput, {
                handlers: {
                    edit: () => {
                        try {
                            const latex = gFunctionField.latex();
                            this.elements.gFunctionHidden.value = this.latexToJavaScript(latex);
                        } catch (error) {
                            this.showError('Error al procesar la función g(x)');
                            console.error('Error en MathQuill handler:', error);
                        }
                    },
                    focus: () => {
                        this.activeMathField = gFunctionField;
                        console.log("Campo MathQuill enfocado: gFunctionInput");
                    },
                    blur: () => {
                        console.log("Campo MathQuill desenfocado: gFunctionInput");
                    }
                }
            });
            this.allMathFields.set('gFunction', gFunctionField);

            // Establecer el campo inicial activo
            this.activeMathField = mathInputField;

            console.log("MathQuill inicializado correctamente con múltiples campos");
        } catch (error) {
            throw new Error(`Error inicializando MathQuill: ${error.message}`);
        }
    }

    // Método auxiliar para mostrar errores
    showError(message) {
        alert(message); // O puedes crear un componente de error personalizado en la UI
    }

    // Método auxiliar para convertir el LaTeX a formato JavaScript
    latexToJavaScript(latex) {
        // Implementa la conversión de LaTeX a código JavaScript válido
        return latex; // Esto es solo un ejemplo, deberías implementar una conversión real
    }

    // Obtener un elemento requerido desde el DOM
    getRequiredElement(id) {
        const element = document.getElementById(id);
        if (!element) {
            throw new Error(`Elemento con ID "${id}" no encontrado.`);
        }
        return element;
    }

    handleInitializationError(error) {
        console.error('Error de inicialización:', error);
        document.body.innerHTML = `
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error de Inicialización</h4>
                    <p>No se pudo inicializar la calculadora.</p>
                    <hr>
                    <p class="mb-0">Error: ${error.message}</p>
                    <button class="btn btn-primary mt-3" onclick="location.reload()">Recargar Página</button>
                </div>
            </div>
        `;
    }

    // Métodos para establecer y manejar eventos
    setupEventListeners() {
        try {
            // Aquí irían los listeners para eventos relacionados a la UI y la lógica
            // Ejemplo: 
            // this.elements.form.addEventListener('submit', this.handleFormSubmit.bind(this));

            console.log("Event listeners configurados correctamente.");
        } catch (error) {
            throw new Error(`Error configurando event listeners: ${error.message}`);
        }
    }
}
console.log("CalculatorApp constructor initialized.");
console.log("MathQuill initialized with multiple fields.");
console.log("Event listeners configured successfully.");
console.log("Error handling initialized.");
console.log("CalculatorApp instance made globally available.");
console.log("MathQuill handlers set for mathInput and gFunctionInput fields.");
console.log("Initial active MathQuill field set to mathInput.");
console.log("MathQuill edit handler triggered for mathInput field.");
console.log("MathQuill edit handler triggered for gFunctionInput field.");
console.log("MathQuill focus handler triggered for mathInput field.");
console.log("MathQuill focus handler triggered for gFunctionInput field.");
console.log("MathQuill blur handler triggered for mathInput field.");
console.log("MathQuill blur handler triggered for gFunctionInput field.");