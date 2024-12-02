class MathQuillManager {
    constructor(elements) {
        this.elements = elements;
        this.allMathFields = new Map();
        this.activeMathField = null;
        this.mathField = null;
        this.mathField = MathQuill.getInterface(2).MathField(document.getElementById('mathField'));
        this.mathQuillFields = [];
    }
    initMathQuill() {
        this.mathQuillFields.forEach(field => {
            // Inicializar MathQuill en el campo
        });
    }
    // Inicializar la entrada matemática
    initializeMathField() {
        this.mathField.latex('');
    }

    // Obtener la entrada en formato latex
    getLatex() {
        return this.mathField.latex();
    }
     // Función para inicializar o limpiar MathQuill según el método seleccionado
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
                        if (!document.activeElement.classList.contains('mathquill-editable')) {
                            this.activeMathField = null;
                        }
                    }, 100);
                });
            }
        } else if (['jacobi', 'gauss_seidel', 'broyden'].includes(selectedMethod)) {
            if (this.mathField) {
                this.mathField = null; // Destruir la instancia si existe
            }
            this.elements.mathInput.innerHTML = ''; // Limpiar contenido
        }
    }

    // Función para agregar dinámicamente nuevos campos MathQuill
    addMathQuillField(elementId, fieldIdentifier) {
        const MQ = MathQuill.getInterface(2);
        const element = document.getElementById(elementId);

        if (element) {
            const mathField = MQ.MathField(element, {
                handlers: {
                    edit: () => {
                        try {
                            const latex = mathField.latex();
                            console.log(`Campo ${fieldIdentifier} editado: ${latex}`);
                        } catch (error) {
                            console.error(`Error en el campo ${fieldIdentifier}:`, error);
                        }
                    }
                }
            });

            this.allMathFields.set(fieldIdentifier, mathField);

            // Gestionar el foco en los campos MathQuill
            element.addEventListener('focus', () => {
                this.activeMathField = mathField;
                console.log(`Campo MathQuill enfocado: ${fieldIdentifier}`);
            });
        }
    }

    // Función para remover un campo MathQuill
    removeMathQuillField(fieldIdentifier) {
        this.allMathFields.delete(fieldIdentifier);
    }

    // Función para obtener el campo MathQuill activo
    getActiveMathField() {
        return this.activeMathField;
    }
    
     // Método para agregar dinámicamente nuevos campos MathQuill
     addMathQuillField(elementId, fieldIdentifier) {
        const MQ = MathQuill.getInterface(2);
        const element = document.getElementById(elementId);

        if (element) {
            const mathField = MQ.MathField(element, {
                handlers: {
                    edit: () => {
                        try {
                            const latex = mathField.latex();
                            console.log(`Campo ${fieldIdentifier} editado: ${latex}`);
                        } catch (error) {
                            console.error(`Error en el campo ${fieldIdentifier}:`, error);
                        }
                    }
                }
            });

            this.allMathFields.set(fieldIdentifier, mathField);

            // Gestionar el foco en los campos MathQuill
            element.addEventListener('focus', () => {
                this.activeMathField = mathField;
                console.log(`Campo MathQuill enfocado: ${fieldIdentifier}`);
            });
        }
    }

    // Método para remover campos MathQuill
    removeMathQuillField(fieldIdentifier) {
        this.allMathFields.delete(fieldIdentifier);
    }

    // Método para obtener el campo activo actual
    getActiveMathField() {
        return this.activeMathField;
    }

    initialize() {
        const MQ = MathQuill.getInterface(2);
        const mathInputField = MQ.MathField(this.elements.mathInput, {
            handlers: {
                edit: () => {
                    try {
                        const latex = mathInputField.latex().trim();
                        this.elements.equationHidden.value = latex;
                    } catch (error) {
                        console.error('Error en MathQuill handler:', error);
                    }
                },
                focus: () => {
                    this.activeMathField = mathInputField;
                }
            }
        });
        this.allMathFields.set('mathInput', mathInputField);

        const gFunctionField = MQ.MathField(this.elements.gFunctionInput, {
            handlers: {
                edit: () => {
                    try {
                        const latex = gFunctionField.latex();
                        this.elements.gFunctionHidden.value = latex;
                    } catch (error) {
                        console.error('Error en MathQuill handler:', error);
                    }
                },
                focus: () => {
                    this.activeMathField = gFunctionField;
                }
            }
        });
        this.allMathFields.set('gFunction', gFunctionField);

        this.activeMathField = mathInputField;
    }

    addMathQuillField(elementId, fieldIdentifier) {
        const MQ = MathQuill.getInterface(2);
        const element = document.getElementById(elementId);
        if (element) {
            const mathField = MQ.MathField(element, {
                handlers: {
                    edit: () => {
                        try {
                            const latex = mathField.latex();
                            console.log(`Campo ${fieldIdentifier} editado: ${latex}`);
                        } catch (error) {
                            console.error(`Error en el campo ${fieldIdentifier}:`, error);
                        }
                    }
                }
            });
            this.allMathFields.set(fieldIdentifier, mathField);
        }
    }

    removeMathQuillField(fieldIdentifier) {
        this.allMathFields.delete(fieldIdentifier);
    }
}
