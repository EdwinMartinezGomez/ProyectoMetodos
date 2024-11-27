class MathFieldManager {
    constructor(uiManager) {
        this.uiManager = uiManager;
        this.activeMathField = null;
        this.allMathFields = new Map();
        this.MQ = MathQuill.getInterface(2);
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