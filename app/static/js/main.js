class CalculatorApp {
    constructor() {
        try {
            this.activeMathField = null;
            this.allMathFields = new Map(); // Mapa para almacenar todos los campos MathQuill
            this.initializeElements();
            this.initializeMathQuill();
            this.setupEventListeners();
            this.uiManager = new UIManager();
            this.mathQuillManager = new MathQuillManager(this.uiManager.elements);
            this.equationManager = new EquationManager(this.uiManager.elements);

            // Hacer la instancia disponible globalmente
            window.calculatorApp = this;
        } catch (error) {
            this.handleInitializationError(error);
        }
    }

    initializeElements() {
        try {
             this.uiManager.initializeElements([
                'calculator-form', 'math-input', 'method', 'equation', 'results', 'resultTable', 
                'equationsContainer', 'variablesContainer', 'find-interval-btn', 'toggle-keyboard-btn', 
                'keyboard-container', 'gFunctionInput', 'gFunctionHidden', 'integrationInputs'
            ]);
            this.elements = {
                form: this.getRequiredElement('calculator-form'),
                mathInput: this.getRequiredElement('math-input'),
                equationHidden: this.getRequiredElement('equation'),
                methodSelect: this.getRequiredElement('method'),
                intervalInputs: this.getRequiredElement('intervalInputs'),
                initialGuessInput: this.getRequiredElement('initialGuessInput'),
                fixedPointInputs: this.getRequiredElement('fixedPointInputs'),
                secantInputs: this.getRequiredElement('secantInputs'),
                systemInputs: this.getRequiredElement('systemInputs'),
                equationsContainer: this.getRequiredElement('equationsContainer'),
                addEquationBtn: this.getRequiredElement('addEquationBtn'),
                variablesContainer: this.getRequiredElement('variablesContainer'),
                resultsDiv: this.getRequiredElement('results'),
                resultTable: this.getRequiredElement('resultTable'),
                plotHtmlContainer: this.getRequiredElement('plotHtmlContainer'),
                findIntervalBtn: this.getRequiredElement('find-interval-btn'),
                initialGuessSystem: this.getRequiredElement('initialGuessSystem'),
                singleEquationInput: this.getRequiredElement('singleEquationInput'),
                // Nuevos elementos añadidos:
                toggleKeyboardBtn: this.getRequiredElement('toggle-keyboard-btn'),
                keyboardContainer: this.getRequiredElement('keyboard-container'),
                gFunctionInput: this.getRequiredElement('gFunctionInput'),
                gFunctionHidden: this.getRequiredElement('gFunctionHidden'),
                integrationInputs: this.getRequiredElement('integrationInputs'), // Añadido
                // Añade referencias a los nuevos campos de integración
                aIntegration: this.getRequiredElement('a_integration'),
                bIntegration: this.getRequiredElement('b_integration'),
                nIntegration: this.getRequiredElement('n_integration'),
                aBisection: this.getRequiredElement('a_bisection'),
                bBisection: this.getRequiredElement('b_bisection'),
                // ... otros elementos según sea necesario
            };
            console.log("Elementos inicializados correctamente:", this.elements);
        } catch (error) {
            throw new Error(`Error inicializando elementos: ${error.message}`);
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
                field.value = ''; // Opcional: limpiar el valor
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
                    setTimeout(() => {
                        if (document.activeElement.classList.contains('mathquill-editable')) {
                            return;
                        }
                        this.activeMathField = null;
                    }, 100);
                });

            }
        } else if (['jacobi', 'gauss_seidel', 'broyden'].includes(selectedMethod)) {
            if (this.mathField) {
                this.mathField = null; // Destruir instancia si existe
            }
            this.elements.mathInput.innerHTML = ''; // Opcionalmente limpiar contenido
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
            <button type="button" class="btn btn-danger removeEquationBtn">Eliminar</button>
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
                },
                focus: () => {
                    this.activeMathField = mathField;
                    console.log(`Campo MathQuill enfocado: equation ${newEquationIndex}`);
                },
                blur: () => {
                    console.log(`Campo MathQuill desenfocado: equation ${newEquationIndex}`);
                }
            }
        });

        this.allMathFields.set(`equation_${newEquationIndex}`, mathField);

        // No es necesario agregar event listeners de focus al elemento DOM
        // Enfocar automáticamente el nuevo campo de ecuación
        mathField.focus();
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
                <button type="button" class="btn btn-danger removeEquationBtn">Eliminar</button>
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
                    },
                    focus: () => {
                        this.activeMathField = mathField;
                        console.log(`Campo MathQuill enfocado: equation ${i}`);
                    },
                    blur: () => {
                        console.log(`Campo MathQuill desenfocado: equation ${i}`);
                    }
                }
            });

            this.allMathFields.set(`equation_${i}`, mathField);
        }

        // **Enfocar automáticamente el primer campo de ecuación**
        const firstEquationInput = equationsList.querySelector('.mathquill-field');
        if (firstEquationInput) {
            firstEquationInput.focus();
        }

        console.log(`Variables actualizadas: ${variablesList.innerHTML}`);
        console.log(`Ecuaciones actualizadas: ${equationsList.innerHTML}`);
    }

    async handleFindInterval() {
        const equation = this.elements.equationHidden.value.trim();

        if (!equation) {
            this.showError('La ecuación es requerida para encontrar el intervalo.');
            return;
        }

        try {
            const response = await fetch('/find_valid_interval', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ equation }) // Solo envía 'equation'
            });

            const data = await response.json();
            if (data.error) {
                this.showError(data.error);
            } else {
                // Actualiza los campos 'a_bisection' y 'b_bisection' con los valores recibidos
                this.elements.aBisection.value = data.a;
                this.elements.bBisection.value = data.b;
            }
        } catch (error) {
            this.showError('Error al comunicarse con el servidor.');
            console.error('Error en handleFindInterval:', error);
        }
    }
}
