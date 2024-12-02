class MathFieldManager {
    constructor(uiManager) {
        this.uiManager = uiManager;
        this.activeMathField = null;
        this.allMathFields = new Map();
        this.MQ = MathQuill.getInterface(2);
        this.mathField = document.getElementById('mathField');
        this.mathFields = [];
    }
     // Método para inicializar el campo de entrada matemático
     initialize() {
        // Inicialización de MathQuill o similar
    }
    initMathFields() {
        this.mathFields.forEach(field => {
            // Inicializar MathQuill en el campo
        });
    }

    getMathFieldValues() {
        // Extraer los valores de los campos
    }

    // Obtener la expresión matemática
    getMathExpression() {
        return this.mathField.value;
    }

    initializeMathField(elementId, hiddenElementId) {
        const element = this.uiManager.getElement(elementId);
        const hiddenElement = this.uiManager.getElement(hiddenElementId);

        const mathField = this.MQ.MathField(element, {
            handlers: {
                edit: () => {
                    try {
                        const latex = mathField.latex().trim();
                        hiddenElement.value = this.latexToJavaScript(latex);
                    } catch (error) {
                        console.error('Error procesando MathQuill:', error);
                    }
                },
                focus: () => {
                    this.activeMathField = mathField;
                },
                blur: () => {
                    this.activeMathField = null;
                }
            }
        });

        this.allMathFields.set(elementId, mathField);
    }
    latexToJavaScript(latex) {
        let processedLatex = latex;
        processedLatex = processedLatex.replace(/\frac\{([^{}]+)\}\{([^{}]+)\}/g, '($1)/($2)');
        processedLatex = processedLatex.replace(/\sqrt\{([^{}]+)\}/g, 'sqrt($1)');
        processedLatex = processedLatex.replace(/\cdot|\times/g, '*');
        processedLatex = processedLatex.replace(/\div/g, '/');
        processedLatex = processedLatex.replace(/\pi/g, 'pi');
        return processedLatex;
    }
}