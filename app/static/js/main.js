class CalculatorApp {
    constructor() {
        try {
            this.activeMathField = null; // Añadido
            this.initializeElements();
            this.initializeMathQuill();
            this.setupEventListeners();
        } catch (error) {
            this.handleInitializationError(error);
        }
    }

    initializeElements() {
        try {
            this.elements = {
                form: this.getRequiredElement('calculator-form'),
                mathInput: this.getRequiredElement('math-input'),
                systemInputs: this.getRequiredElement('systemInputs'),
                equationHidden: this.getRequiredElement('equation'),
                methodSelect: this.getRequiredElement('method'),
                intervalInputs: this.getRequiredElement('intervalInputs'),
                initialGuessInput: this.getRequiredElement('initialGuessInput'),
                fixedPointInputs: this.getRequiredElement('fixedPointInputs'),
                secantInputs: this.getRequiredElement('secantInputs'), // Añadido
                systemInputs: this.getRequiredElement('systemInputs'), // Añadido
                equationsContainer: this.getRequiredElement('equationsContainer'), // Añadido
                addEquationBtn: this.getRequiredElement('addEquationBtn'), // Añadido
                variablesContainer: this.getRequiredElement('variablesContainer'), // Añadido
                resultsDiv: this.getRequiredElement('results'),
                resultTable: this.getRequiredElement('resultTable'),
                plotHtmlContainer: this.getRequiredElement('plotHtmlContainer'),
                findIntervalBtn: this.getRequiredElement('find-interval-btn'),
                initialGuessSystem: this.getRequiredElement('initialGuessSystem'), // Añadido
                singleEquationInput: this.getRequiredElement('singleEquationInput'), // **Añadido**
                current_input: this.getRequiredElement('current_input'),// **Añadido**


            };
        } catch (error) {
            throw new Error(`Error inicializando elementos: ${error.message}`);
        }
    }

    getRequiredElement(id) {
        const element = document.getElementById(id);
        if (!element) {
            throw new Error(`Elemento requerido no encontrado: ${id}`);
        }
        return element;
    }

    // Dentro de la clase CalculatorApp

    initializeMathQuill() {
        try {
            if (typeof MathQuill === 'undefined') {
                throw new Error('MathQuill no está disponible');
            }

            const MQ = MathQuill.getInterface(2);

            // Inicializar MathQuill para el campo estático
            this.mathField = MQ.MathField(this.elements.mathInput, {
                handlers: {
                    edit: () => {
                        try {
                            const latex = this.mathField.latex();
                            this.elements.equationHidden.value = this.latexToJavaScript(latex);
                        } catch (error) {
                            this.showError('Error al procesar la ecuación matemática');
                            console.error('Error en MathQuill handler:', error);
                        }
                    }
                }
            });

            // Rastrear el campo activo
            this.elements.mathInput.addEventListener('focus', () => {
                this.activeMathField = this.mathField;
            });

            this.elements.mathInput.addEventListener('blur', () => {
                this.activeMathField = null;
            });

        } catch (error) {
            throw new Error(`Error inicializando MathQuill: ${error.message}`);
        }
    }




    setupEventListeners() {
        try {
            this.elements.form.addEventListener('submit', this.handleFormSubmit.bind(this));
            this.elements.methodSelect.addEventListener('change', this.handleMethodChange.bind(this));
            this.elements.findIntervalBtn.addEventListener('click', this.handleFindInterval.bind(this));

            // Añadir listeners para los nuevos botones
            this.elements.addEquationBtn.addEventListener('click', this.addEquationField.bind(this));
            const numVariablesInput = this.elements.form.querySelector('#numVariables');
            if (numVariablesInput) {
                numVariablesInput.addEventListener('change', this.updateVariables.bind(this));
                // Agregar listener para la tecla Enter en #numVariables
                numVariablesInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault(); // Evita que el formulario se envíe si está dentro de uno
                        this.updateVariables();
                    }
                });
            }

            // Validaciones en tiempo real para initial_guess_system
            const initialGuessSystemInput = this.elements.initialGuessSystem.querySelector('input');
            if (initialGuessSystemInput) {
                initialGuessSystemInput.addEventListener('input', this.validateInitialGuessSystem.bind(this));
            }

            // Inicializar el estado correcto al cargar la página
            this.handleMethodChange({ target: this.elements.methodSelect });
        } catch (error) {
            throw new Error(`Error configurando event listeners: ${error.message}`);
        }
    }



    validateInitialGuessSystem(event) {
        const input = event.target;
        const pattern = /^-?\d+(\.\d+)?(?:\s*,\s*-?\d+(\.\d+)?)*$/;
        if (!pattern.test(input.value)) {
            input.setCustomValidity('Ingrese valores numéricos separados por comas, por ejemplo: 1,1');
        } else {
            input.setCustomValidity('');
        }
    }
    handleMethodChange(event) {
        const selectedMethod = event.target.value;
    
        // Reset 'required' for all fields
        this.elements.form.querySelectorAll('input').forEach(input => input.required = false);
    
        // Show/hide fields based on the selected method
        if (selectedMethod === 'bisection') {
            this.elements.intervalInputs.style.display = 'flex';
            this.elements.findIntervalBtn.style.display = 'block';
            this.elements.form.querySelector('#a').required = true;
            this.elements.form.querySelector('#b').required = true;
    
            this.hideFields(['secantInputs', 'initialGuessInput', 'fixedPointInputs', 
                            'systemInputs', 'equationsContainer', 'variablesContainer', 
                            'initialGuessSystem']);
            this.elements.singleEquationInput.style.display = 'block';
            this.enableFields(['a', 'b']);
        } 
        else if (selectedMethod === 'secant') {
            this.elements.secantInputs.style.display = 'flex';
            this.elements.secantInputs.querySelectorAll('input').forEach(input => input.required = true);
    
            this.hideFields(['intervalInputs', 'findIntervalBtn', 'initialGuessInput', 
                            'fixedPointInputs', 'systemInputs', 'equationsContainer', 
                            'variablesContainer', 'initialGuessSystem']);
            this.elements.singleEquationInput.style.display = 'block';
            this.disableFields(['a', 'b']);
        } 
        else if (selectedMethod === 'newton') {
            this.elements.initialGuessInput.style.display = 'block';
            this.elements.initialGuessInput.querySelector('input').required = true;
    
            this.hideFields(['intervalInputs', 'findIntervalBtn', 'secantInputs', 
                            'fixedPointInputs', 'systemInputs', 'equationsContainer', 
                            'variablesContainer', 'initialGuessSystem']);
            this.elements.singleEquationInput.style.display = 'block';
            this.disableFields(['a', 'b']);
        } 
        else if (selectedMethod === 'fixed_point') {
            this.elements.initialGuessInput.style.display = 'block';
            this.elements.fixedPointInputs.style.display = 'block';
            this.elements.initialGuessInput.querySelector('input').required = true;
            this.elements.fixedPointInputs.querySelector('#gFunction').required = true;
    
            this.hideFields(['intervalInputs', 'findIntervalBtn', 'secantInputs', 
                            'systemInputs', 'equationsContainer', 'variablesContainer', 
                            'initialGuessSystem']);
            this.elements.singleEquationInput.style.display = 'block';
            this.disableFields(['a', 'b']);
        } 
        else if (selectedMethod === 'jacobi' || selectedMethod === 'gauss_seidel') {
            this.elements.systemInputs.style.display = 'block';
            this.elements.equationsContainer.style.display = 'block';
            this.elements.variablesContainer.style.display = 'block';
            this.elements.initialGuessSystem.style.display = 'block';
            this.elements.initialGuessSystem.querySelector('input').required = true;
    
            this.hideFields(['intervalInputs', 'findIntervalBtn', 'secantInputs', 
                            'initialGuessInput', 'fixedPointInputs', 'singleEquationInput']);
            this.disableFields(['a', 'b']);
    
            // Add 'required' to all 'variables[]' fields dynamically
            const variableInputs = this.elements.variablesContainer.querySelectorAll('input[name="variables[]"]');
            variableInputs.forEach(input => input.required = true);
        } 
        else {
            this.hideFields(['intervalInputs', 'findIntervalBtn', 'initialGuessInput', 
                            'fixedPointInputs', 'secantInputs', 'systemInputs', 
                            'equationsContainer', 'variablesContainer', 'initialGuessSystem']);
            this.elements.singleEquationInput.style.display = 'block';
            this.disableFields(['a', 'b']);
        }
    
        // Initialize or clear MathQuill based on the selected method
        this.toggleMathQuill(selectedMethod);
    
        // Optional: Update example in MathQuill
        const examples = {
            'bisection': 'x^2 - 4',
            'newton': 'x^3 - 2x - 5',
            'secant': '',
            'fixed_point': '',
            'jacobi': 'Sistema de 2 ecuaciones',
            'gauss_seidel': 'Sistema de 2 ecuaciones'
        };
        if (this.activeMathField && examples[selectedMethod]) {
            this.activeMathField.latex(examples[selectedMethod]);
        } else if (this.activeMathField) {
            this.activeMathField.latex('');
        }
    }
    
    // Utility functions to show/hide fields and enable/disable inputs
    hideFields(fields) {
        fields.forEach(field => this.elements[field].style.display = 'none');
    }
    
    enableFields(ids) {
        ids.forEach(id => {
            const field = this.elements.form.querySelector(`#${id}`);
            if (field) {
                field.disabled = false;
            }
        });
    }
    
    disableFields(ids) {
        ids.forEach(id => {
            const field = this.elements.form.querySelector(`#${id}`);
            if (field) {
                field.disabled = true;
                field.value = ''; // Optional: clear the value
            }
        });
    }
    
    // Function to initialize or clear MathQuill
    toggleMathQuill(selectedMethod) {
        if (['bisection', 'newton', 'secant', 'fixed_point'].includes(selectedMethod)) {
            if (!this.mathField && this.elements.mathInput.offsetParent !== null) {
                const MQ = MathQuill.getInterface(2);
                this.mathField = MQ.MathField(this.elements.mathInput, {
                    handlers: {
                        edit: () => {
                            try {
                                const latex = this.mathField.latex();
                                this.elements.equationHidden.value = this.latexToJavaScript(latex);
                            } catch (error) {
                                this.showError('Error al procesar la ecuación matemática');
                                console.error('Error en MathQuill handler:', error);
                            }
                        }
                    }
                });
                this.elements.mathInput.addEventListener('focus', () => {
                    this.activeMathField = this.mathField;
                });
                this.elements.mathInput.addEventListener('blur', () => {
                    this.activeMathField = null;
                });
            }
        } else if (['jacobi', 'gauss_seidel'].includes(selectedMethod)) {
            if (this.mathField) {
                this.mathField = null; // Destroy instance if exists
            }
            this.elements.mathInput.innerHTML = ''; // Optionally clear content
        }
    }
    
    addEquationField() {
        const numVariablesInput = this.elements.form.querySelector('#numVariables');
        let numVariables = parseInt(numVariablesInput.value, 10);

        // Verificar si se ha alcanzado el máximo de variables
        if (numVariables >= 10) {
            this.showError('El número máximo de variables es 10.');
            return;
        }

        // Incrementar el número de variables
        numVariables += 1;
        numVariablesInput.value = numVariables;

        // Actualizar los campos de variables y ecuaciones
        this.updateVariables();

        // Agregar una nueva ecuación
        const equationList = this.elements.equationsContainer.querySelector('#equationsList');
        const currentEquations = equationList.querySelectorAll('.equation-input').length;
        const newEquationIndex = currentEquations + 1;

        const equationDiv = document.createElement('div');
        equationDiv.className = 'input-group mb-2 equation-input';
        equationDiv.innerHTML = `
        <span class="input-group-text">Ecuación ${newEquationIndex}:</span>
        <div class="mathquill-field form-control" id="mathquill_equation_${newEquationIndex}"></div>
        <input type="hidden" name="equations[]" id="equation_${newEquationIndex}">
    `;
        equationList.appendChild(equationDiv);

        // Inicializar MathQuill en el nuevo campo de ecuación
        const mathQuillDiv = equationDiv.querySelector('.mathquill-field');
        const equationHiddenInput = equationDiv.querySelector(`#equation_${newEquationIndex}`);

        const MQ = MathQuill.getInterface(2);
        const mathField = MQ.MathField(mathQuillDiv, {
            handlers: {
                edit: () => {
                    try {
                        const latex = mathField.latex();
                        equationHiddenInput.value = this.latexToJavaScript(latex);
                    } catch (error) {
                        this.showError('Error al procesar la ecuación matemática');
                        console.error('Error en MathQuill handler:', error);
                    }
                }
            }
        });

        // Agregar event listeners para rastrear el campo activo
        mathQuillDiv.addEventListener('focus', () => {
            this.activeMathField = mathField;
        });

        mathQuillDiv.addEventListener('blur', () => {
            this.activeMathField = null;
        });

        // Opcional: Enfocar automáticamente el nuevo campo de ecuación
        mathQuillDiv.focus();
    }

    updateVariables() {
        const numVariables = parseInt(this.elements.form.querySelector('#numVariables').value, 10);
        const variablesList = this.elements.variablesContainer.querySelector('#variablesList');
        variablesList.innerHTML = ''; // Limpiar variables existentes

        console.log(`Actualizando variables: número de variables = ${numVariables}`);

        for (let i = 1; i <= numVariables; i++) {
            const varDiv = document.createElement('div');
            varDiv.className = 'input-group mb-2';
            varDiv.innerHTML = `
            <span class="input-group-text">Variable ${i}:</span>
            <input type="text" class="form-control" name="variables[]" placeholder="Ingrese la variable ${i}" required pattern="[a-zA-Z]+">
        `;
            variablesList.appendChild(varDiv);
        }

        // Limpiar ecuaciones existentes y agregar según el número de variables
        const equationsList = this.elements.equationsContainer.querySelector('#equationsList');
        equationsList.innerHTML = '';
        for (let i = 1; i <= numVariables; i++) {
            const equationDiv = document.createElement('div');
            equationDiv.className = 'input-group mb-2 equation-input';
            equationDiv.innerHTML = `
            <span class="input-group-text">Ecuación ${i}:</span>
            <div class="mathquill-field form-control" id="mathquill_equation_${i}"></div>
            <input type="hidden" name="equations[]" id="equation_${i}">
        `;
            equationsList.appendChild(equationDiv);

            // Inicializar MathQuill en el campo de ecuación
            const mathQuillDiv = equationDiv.querySelector('.mathquill-field');
            const equationHiddenInput = equationDiv.querySelector(`#equation_${i}`);

            const MQ = MathQuill.getInterface(2);
            const mathField = MQ.MathField(mathQuillDiv, {
                handlers: {
                    edit: () => {
                        try {
                            const latex = mathField.latex();
                            equationHiddenInput.value = this.latexToJavaScript(latex);
                        } catch (error) {
                            this.showError('Error al procesar la ecuación matemática');
                            console.error('Error en MathQuill handler:', error);
                        }
                    }
                }
            });

            // Agregar event listeners para rastrear el campo activo
            mathQuillDiv.addEventListener('focus', () => {
                this.activeMathField = mathField;
            });

            mathQuillDiv.addEventListener('blur', () => {
                this.activeMathField = null;
            });
        }

        // **Enfocar automáticamente el primer campo de ecuación**
        const firstEquationInput = equationsList.querySelector('.mathquill-field');
        if (firstEquationInput) {
            firstEquationInput.focus();
        }

        console.log(`Variables actualizadas: ${variablesList.innerHTML}`);
        console.log(`Ecuaciones actualizadas: ${equationsList.innerHTML}`);
    }




    async handleFindInterval(event) {
        event.preventDefault();
        this.clearErrors();

        try {
            const equation = this.elements.equationHidden.value.trim();
            if (!equation) {
                throw new Error('La ecuación es requerida para encontrar un intervalo.');
            }

            const response = await fetch('/find_valid_interval', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ equation })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Error al encontrar intervalo.');
            }

            const data = await response.json();
            this.elements.form.a.value = data.a;
            this.elements.form.b.value = data.b;

            // Mostrar mensaje de éxito (opcional)
            alert(`Intervalo encontrado: a = ${data.a}, b = ${data.b}`);
        } catch (error) {
            this.showError(error.message);
        }
    }

    showSuccess(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'alert alert-success alert-dismissible fade show mt-3';
        successDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        this.elements.form.insertBefore(successDiv, this.elements.form.firstChild);
    }

    validateNumberInput(event) {
        const input = event.target;
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);

        if (isNaN(value)) {
            input.setCustomValidity('Por favor ingrese un número válido');
        } else if (min !== undefined && value < min) {
            input.setCustomValidity(`El valor mínimo permitido es ${min}`);
        } else if (max !== undefined && value > max) {
            input.setCustomValidity(`El valor máximo permitido es ${max}`);
        } else {
            input.setCustomValidity('');
        }
    }

    async handleFormSubmit(event) {
        event.preventDefault();
        this.clearErrors();
        this.showLoading();
    
        try {
            const formData = await this.validateAndPrepareFormData();
            console.log("Datos de formulario validados:", formData); // Asegúrate de que las ecuaciones y variables sean correctas
    
            const response = await this.sendCalculationRequest(formData);
            await this.handleCalculationResponse(response);
        } catch (error) {
            this.handleError(error);
        } finally {
            this.hideLoading();
        }
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

    clearErrors() {
        const existingErrors = document.querySelectorAll('.alert-danger');
        existingErrors.forEach(error => error.remove());

        // También limpiar el div de resultados y la gráfica
        this.elements.resultTable.innerHTML = '';
        this.elements.plotHtmlContainer.innerHTML = '';
        this.elements.resultsDiv.style.display = 'none';
    }

    async sendCalculationRequest(formData) {
        try {
            // Si es un sistema, enviar como tal
            if (formData.equations && formData.variables) {
                // Convertir las estimaciones iniciales a una lista de números
                formData.initial_guess = formData.initial_guess;
            }

            const response = await fetch('/calculate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.error || `Error del servidor: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            if (error instanceof TypeError && error.message.includes('fetch')) {
                throw new Error('No se pudo conectar con el servidor. Por favor, verifique su conexión.');
            }
            throw error;
        }
    }
    async handleCalculationResponse(response) {
        if (response.error) {
            this.showError(response.error);
            return;
        }

        try {
            if (response.solution) {
                // Mostrar resultados para sistemas
                this.displaySystemResults(response);
            } else if (response.root) {
                // Mostrar resultados para métodos de una sola ecuación
                this.displayMainResults(response);
            }

            // Verificar y mostrar el historial de iteraciones si está presente
            if (response.iteration_history) {
                if (response.solution) {
                    // Método para sistemas de ecuaciones
                    this.displayIterationHistory(response.iteration_history, response.solution);
                } else {
                    // Método para una sola ecuación
                    this.displayIterationHistory(response.iteration_history, response.root);
                }
            }

            if (response.plot_json) {
                this.renderPlot(response.plot_json);
            } else {
                console.warn('No se proporcionó plot_json en la respuesta.');
            }
        } catch (error) {
            this.showError(`Error al mostrar resultados: ${error.message}`);
        }

        this.elements.resultsDiv.style.display = 'block';
    }



    renderPlot(plotJson) {
        try {
            const plotData = JSON.parse(plotJson);

            // Renderizar la gráfica inicial con datos y layout
            Plotly.newPlot('plotHtmlContainer', plotData.data, plotData.layout).then(() => {
                if (plotData.frames && plotData.frames.length > 0) {
                    // Añadir los frames a la gráfica
                    Plotly.addFrames('plotHtmlContainer', plotData.frames);

                    // Opcional: Iniciar la animación automáticamente
                    Plotly.animate('plotHtmlContainer', {
                        transition: {
                            duration: 700,  // Ajusta la duración según prefieras
                            easing: 'linear'
                        },
                        frame: {
                            duration: 700,
                            redraw: false
                        }
                    });
                }
            }).catch(error => {
                console.error('Error al renderizar la gráfica:', error);
                this.showError('No se pudo renderizar la gráfica correctamente.');
            });
        } catch (error) {
            console.error('Error al parsear plot_json:', error);
            this.showError('Error al procesar los datos de la gráfica.');
        }
    }

    handleError(error) {
        console.error('Error en la aplicación:', error);

        let userMessage = 'Ha ocurrido un error inesperado.';
        if (error.message.includes('servidor')) {
            userMessage = error.message;
        } else if (error.message.includes('gráfico')) {
            userMessage = 'No se pudo generar el gráfico. Los resultados numéricos están disponibles.';
        } else if (error.message.includes('ecuación')) {
            userMessage = 'La ecuación ingresada no es válida. Por favor, verifíquela.';
        }

        this.showError(userMessage);
    }

    showError(message) {
        // Limpiar contenedores
        this.elements.resultTable.innerHTML = '';
        this.elements.plotHtmlContainer.innerHTML = '';
        this.elements.resultsDiv.style.display = 'none';

        // Mostrar mensaje de error
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
        errorDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        this.elements.form.insertBefore(errorDiv, this.elements.form.firstChild);
    }

    handleInitializationError(error) {
        console.error('Error de inicialización:', error);
        document.body.innerHTML = `
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error de Inicialización</h4>
                    <p>Lo sentimos, no se pudo inicializar la calculadora.</p>
                    <hr>
                    <p class="mb-0">Error: ${error.message}</p>
                    <button class="btn btn-primary mt-3" onclick="location.reload()">Recargar Página</button>
                </div>
            </div>
        `;
    }
    async validateAndPrepareFormData() {
        const method = this.elements.methodSelect.value;
        const iterations = parseInt(this.elements.form.iterations.value, 10);
    
        if (!method) {
            throw new Error('Debe seleccionar un método numérico');
        }
        if (isNaN(iterations) || iterations < 1 || iterations > 1000) {
            throw new Error('El número de iteraciones debe estar entre 1 y 1000');
        }
    
        let formData = { method, iterations };
    
        if (['jacobi', 'gauss_seidel'].includes(method)) {
            // Obtener ecuaciones y variables
            const equations = Array.from(this.elements.form.querySelectorAll('input[name="equations[]"]')).map(input => input.value.trim());
            const variables = Array.from(this.elements.form.querySelectorAll('input[name="variables[]"]')).map(input => input.value.trim());
    
            if (!equations.length || !variables.length) {
                throw new Error('Debe ingresar al menos una ecuación y una variable.');
            }
            if (equations.length !== variables.length) {
                throw new Error('El número de ecuaciones debe ser igual al número de variables.');
            }
    
            // Validar que las variables no estén vacías y cumplan el patrón
            for (const variable of variables) {
                if (!variable || !/^[a-zA-Z]+$/.test(variable)) {
                    throw new Error('Cada variable debe contener letras válidas.');
                }
            }
    
            formData.equations = equations;
            formData.variables = variables;
    
            // Validar el punto inicial
            const initial_guess_str = this.elements.initialGuessSystem.querySelector('input').value.trim();
            if (!initial_guess_str) {
                throw new Error('El punto inicial es requerido para métodos de sistemas.');
            }
    
            const initial_guess = initial_guess_str.split(',').map(val => parseFloat(val.trim()));
            if (initial_guess.length !== variables.length) {
                throw new Error('El punto inicial debe tener el mismo número de elementos que variables.');
            }
    
            formData.initial_guess = initial_guess;
        } else {
            // Métodos para una sola ecuación
            const equation = this.elements.equationHidden.value.trim();
            const method = this.elements.methodSelect.value;

            if (!equation) {
                throw new Error('La ecuación es requerida');
            }

            formData.equation = equation;

            if (method === 'bisection') {
                // Validar los límites del intervalo para Bisección
                const a = parseFloat(this.elements.form.a.value);
                const b = parseFloat(this.elements.form.b.value);

                if (isNaN(a) || isNaN(b)) {
                    throw new Error('Los límites del intervalo deben ser números válidos');
                }
                if (a >= b) {
                    throw new Error('El límite inferior (a) debe ser menor que el superior (b)');
                }

                formData.a = a;
                formData.b = b;
            } else if (method === 'secant') {
                // Validar las estimaciones iniciales para Secante
                const x0 = parseFloat(this.elements.form.x0.value);
                const x1 = parseFloat(this.elements.form.x1.value);

                if (isNaN(x0) || isNaN(x1)) {
                    throw new Error('Las estimaciones iniciales x₀ y x₁ deben ser números válidos');
                }
                if (x0 === x1) {
                    throw new Error('Las estimaciones iniciales x₀ y x₁ deben ser distintas');
                }

                formData.x0 = x0;
                formData.x1 = x1;
            } else if (method === 'newton' || method === 'fixed_point') {
                // Validar punto inicial para Newton-Raphson y Punto Fijo
                const initial_guess = parseFloat(this.elements.form.initial_guess.value);

                if (isNaN(initial_guess)) {
                    throw new Error('El punto inicial debe ser un número válido');
                }

                formData.initial_guess = initial_guess;

                if (method === 'fixed_point') {
                    const gFunction = this.elements.fixedPointInputs.querySelector('#gFunction').value.trim();
                    if (!gFunction) {
                        throw new Error('La función g(x) es requerida para el método de Punto Fijo');
                    }
                    formData.gFunction = gFunction;
                }
            }
        }

        return formData;
    }


    replaceFractions(latex) {
        let startIndex = latex.indexOf('\\frac');
        while (startIndex !== -1) {
            let numerator = '';
            let denominator = '';
            let index = startIndex + 5; // Saltar '\\frac'
            let braceCount = 0;

            // Obtener el numerador
            if (latex[index] === '{') {
                index++;
                braceCount++;
                let numStart = index;
                while (braceCount > 0 && index < latex.length) {
                    if (latex[index] === '{') braceCount++;
                    else if (latex[index] === '}') braceCount--;
                    index++;
                }
                numerator = latex.substring(numStart, index - 1);
            } else {
                throw new Error("Sintaxis inválida en \\frac");
            }

            // Obtener el denominador
            if (latex[index] === '{') {
                index++;
                braceCount = 1;
                let denStart = index;
                while (braceCount > 0 && index < latex.length) {
                    if (latex[index] === '{') braceCount++;
                    else if (latex[index] === '}') braceCount--;
                    index++;
                }
                denominator = latex.substring(denStart, index - 1);
            } else {
                throw new Error("Sintaxis inválida en \\frac");
            }

            // Reemplazar la fracción en el LaTeX original
            let fracContent = latex.substring(startIndex, index);
            let replacedContent = `(${this.replaceFractions(numerator)})/(${this.replaceFractions(denominator)})`;
            latex = latex.replace(fracContent, replacedContent);

            // Buscar la siguiente fracción
            startIndex = latex.indexOf('\\frac');
        }
        return latex;
    }

    latexToJavaScript(latex) {
        let processedLatex = latex;
        console.log("Original LaTeX:", processedLatex);
       
        // Eliminar caracteres de control y espacios invisibles
        processedLatex = processedLatex.replace(/\u200b/g, '');
        
        // Reemplazar fracciones recursivamente
        try {
            processedLatex = this.replaceFractions(processedLatex);
        } catch (error) {
            console.error("Error al reemplazar fracciones:", error);
            alert(error.message);
            throw error;
        }
        
        console.log("Después de reemplazar fracciones:", processedLatex);
    
        // Dividir en LHS y RHS
        const [lhs, rhs] = processedLatex.split('=');
        if (!lhs || !rhs) {
            throw new Error("La ecuación debe contener un solo signo de igual '='.");
        }
    
        // Evaluar RHS para convertir fracciones a decimales si es necesario
        let rhsEvaluated;
        try {
            rhsEvaluated = eval(rhs.trim()); // Convierte fracciones como '1/2' a decimales
        } catch (error) {
            throw new Error("El lado derecho de la ecuación debe ser un número o una fracción.");
        }
    
        // Validar que RHS sea numérico después de la evaluación
        if (isNaN(rhsEvaluated)) {
            throw new Error("El lado derecho de la ecuación debe ser un número.");
        }
    
        // Continuar con las transformaciones y verificar el estado
        processedLatex = processedLatex.replace(/Math\./g, '');
        console.log("Después de eliminar 'Math.':", processedLatex);
    
        // Reemplazar otros comandos matemáticos
        processedLatex = processedLatex.replace(/\\sqrt\{([^{}]+)\}/g, 'sqrt($1)');
        processedLatex = processedLatex.replace(/\\left|\\right/g, '');
        processedLatex = processedLatex.replace(/\\cdot|\\times/g, '*');
        processedLatex = processedLatex.replace(/\\div/g, '/');
        processedLatex = processedLatex.replace(/\\pi/g, 'pi');
        processedLatex = processedLatex.replace(/\\ln/g, 'log');
        processedLatex = processedLatex.replace(/\\log/g, 'log10');
        processedLatex = processedLatex.replace(/\\exp\{([^{}]+)\}/g, 'exp($1)');
        processedLatex = processedLatex.replace(/\\exp/g, 'exp');
        processedLatex = processedLatex.replace(/\\sin/g, 'sin');
        processedLatex = processedLatex.replace(/\\cos/g, 'cos');
        processedLatex = processedLatex.replace(/\\tan/g, 'tan');
    
        // Reemplazar exponentes y verificar estado
        processedLatex = processedLatex.replace(/e\^\{([^}]+)\}/g, 'exp($1)');
        processedLatex = processedLatex.replace(/\^/g, '**');
        console.log("Después de reemplazar exponentes:", processedLatex);
    
        // Reglas de multiplicación implícita ajustadas y estado final
        processedLatex = processedLatex.replace(/(\d)([a-zA-Z(])/g, '$1*$2');
        processedLatex = processedLatex.replace(/([a-zA-Z])(\d)/g, '$1*$2');
        processedLatex = processedLatex.replace(/(\d)\(/g, '$1*(');
        processedLatex = processedLatex.replace(/([a-zA-Z])\(/g, '$1*(');
        processedLatex = processedLatex.replace(/\)([a-zA-Z])/g, ')*$1');
        processedLatex = processedLatex.replace(/\)(\d)/g, ')*$1');
        
        // Validar que los paréntesis estén equilibrados
        const openParens = (processedLatex.match(/\(/g) || []).length;
        const closeParens = (processedLatex.match(/\)/g) || []).length;
        if (openParens !== closeParens) {
            throw new Error("La ecuación contiene paréntesis desbalanceados.");
        }
    
        // Eliminar espacios
        processedLatex = processedLatex.replace(/\s+/g, '');
        
        console.log("Procesado LaTeX a JavaScript:", processedLatex);
        return processedLatex;
    }
    
    
    displayResults(result) {
        this.elements.resultTable.innerHTML = '';

        if (result.solution) {
            // Mostrar la solución
            const solution = result.solution;
            const table = document.createElement('table');
            table.className = 'table table-bordered';

            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            const varHeader = document.createElement('th');
            varHeader.textContent = 'Variable';
            const valHeader = document.createElement('th');
            valHeader.textContent = 'Valor';
            headerRow.appendChild(varHeader);
            headerRow.appendChild(valHeader);
            thead.appendChild(headerRow);
            table.appendChild(thead);

            const tbody = document.createElement('tbody');

            for (const [varName, value] of Object.entries(solution)) {
                const row = document.createElement('tr');
                const varCell = document.createElement('td');
                varCell.textContent = varName;
                const valCell = document.createElement('td');
                valCell.textContent = value.toFixed(6);
                row.appendChild(varCell);
                row.appendChild(valCell);
                tbody.appendChild(row);
            }

            table.appendChild(tbody);
            this.elements.resultTable.appendChild(table);
        }

        // Puedes agregar más detalles según lo necesites
    }

    displaySystemResults(result) {
        const solution = result.solution;
        const keys = Object.keys(solution);
        const values = Object.values(solution);

        // Mostrar si convergió
        this.addResultRow(
            this.elements.resultTable,
            'Convergió',
            result.converged ? 'Sí' : 'No'
        );

        // Mostrar número de iteraciones
        this.addResultRow(
            this.elements.resultTable,
            'Iteraciones realizadas',
            result.iterations
        );

        // Mostrar las soluciones
        keys.forEach((varName, index) => {
            this.addResultRow(
                this.elements.resultTable,
                `Solución para ${varName}`,
                values[index]
            );
        });
    }

    displayMainResults(result) {
        const keysToShow = ['converged', 'iterations', 'root'];
        const mainResults = Object.entries(result)
            .filter(([key]) => keysToShow.includes(key))
            .forEach(([key, value]) => {
                this.addResultRow(
                    this.elements.resultTable,
                    this.formatters.key(key),
                    this.formatters.value(value)
                );
            });
    }

    displayIterationHistory(history, solution = null) {
        const iterationTable = this.createIterationTable(history, solution);
        this.elements.resultTable.appendChild(iterationTable);
    }


    addResultRow(table, label, value) {
        const row = document.createElement('tr');
        const labelCell = document.createElement('td');
        const valueCell = document.createElement('td');

        labelCell.textContent = label;
        valueCell.textContent = value;
        labelCell.className = 'fw-bold';

        row.append(labelCell, valueCell);
        table.appendChild(row);
    }

    createIterationTable(iterations, solution = null) {
        const table = document.createElement('table');
        table.className = 'table table-striped table-bordered mt-3';

        let headers = ['Iteración'];
        if (solution && typeof solution === 'object' && !Array.isArray(solution)) {
            // Método para sistemas de ecuaciones
            const vars = Object.keys(solution);
            headers = headers.concat(vars, 'Error');
        } else {
            // Método para una sola ecuación
            headers = headers.concat(['x', 'f(x)', 'Error']);
        }

        table.appendChild(this.createTableHeader(headers));
        table.appendChild(this.createTableBody(iterations, solution));

        return table;
    }

    createTableHeader(headers) {
        const thead = document.createElement('thead');
        const row = document.createElement('tr');
        headers.forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            th.className = 'text-center';
            row.appendChild(th);
        });
        thead.appendChild(row);
        return thead;
    }

    createTableBody(iterations, solution = null) {
        const tbody = document.createElement('tbody');
        iterations.forEach((iter) => {
            const row = document.createElement('tr');
            const cells = [];

            // Iteración
            const iterCell = document.createElement('td');
            iterCell.textContent = iter.iteration;
            iterCell.className = 'text-center';
            cells.push(iterCell);

            if (solution && typeof solution === 'object' && !Array.isArray(solution)) {
                // Método para sistemas de ecuaciones
                iter.x.forEach(val => {
                    const valCell = document.createElement('td');
                    valCell.textContent = val.toFixed(6);
                    valCell.className = 'text-center';
                    cells.push(valCell);
                });
                // Error
                const errorCell = document.createElement('td');
                errorCell.textContent = iter.error !== undefined ? iter.error.toFixed(6) : '-';
                errorCell.className = 'text-center';
                cells.push(errorCell);
            } else {
                // Método para una sola ecuación
                // x
                const xCell = document.createElement('td');
                xCell.textContent = iter.x.toFixed(6);
                xCell.className = 'text-center';
                cells.push(xCell);

                // f(x)
                const fxCell = document.createElement('td');
                fxCell.textContent = iter.fx !== undefined ? iter.fx.toFixed(6) : '-';
                fxCell.className = 'text-center';
                cells.push(fxCell);

                // Error
                const errorCell = document.createElement('td');
                errorCell.textContent = iter.error !== undefined ? iter.error.toFixed(6) : '-';
                errorCell.className = 'text-center';
                cells.push(errorCell);
            }

            // Añadir las celdas a la fila
            cells.forEach(cell => row.appendChild(cell));
            tbody.appendChild(row);
        });
        return tbody;
    }

    formatters = {
        key: (key) => {
            const formatMap = {
                'root': 'Raíz encontrada',
                'iterations': 'Iteraciones realizadas',
                'converged': 'Converge'
            };
            return formatMap[key] || key;
        },
        value: (value) => {
            if (value == null) return 'N/A';
            if (typeof value === 'number') {
                if (value === 0) {
                    return '0';
                } else if (Math.abs(value) < 0.0001 || Math.abs(value) > 9999) {
                    return value.toExponential(4);
                } else {
                    return value.toFixed(6);
                }
            }
            if (typeof value === 'boolean') return value ? 'Sí' : 'No';
            return String(value);
        }
    };

    // Función para manejar métodos de sistemas (opcional: agregar representaciones gráficas)
    // Puedes implementar visualizaciones específicas para sistemas si lo deseas.

}
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.calculatorApp = new CalculatorApp();
    } catch (error) {
        console.error('Error fatal al inicializar la aplicación:', error);
        document.body.innerHTML = `
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error Fatal</h4>
                    <p>No se pudo iniciar la aplicación. Por favor, recargue la página o contacte al soporte técnico.</p>
                    <button class="btn btn-primary mt-3" onclick="location.reload()">Recargar Página</button>
                </div>
            </div>
        `;
    }
});
