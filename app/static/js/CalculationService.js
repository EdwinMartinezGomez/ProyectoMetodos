class CalculationService {
    constructor() {
        // Puedes inyectar dependencias si es necesario
    }
    async sendCalculationRequest(formData) {
        try {
            if (formData.equations && formData.variables) {
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

     // Reemplazar fracciones en la ecuación
     replaceFractions(eq) {
        while (eq.includes('\\frac')) {
            const match = eq.match(/\\frac\{([^{}]+)\}\{([^{}]+)\}/);
            if (!match) break;

            const [fullMatch, numerator, denominator] = match;
            const replacement = `(${numerator})/(${denominator})`;

            eq = eq.replace(fullMatch, replacement);
        }
        return eq;
    }

    // Convertir LaTeX a JavaScript
    latexToJavaScript(latex) {
        let processedLatex = latex;
        console.log("Original LaTeX:", processedLatex);

        // Reemplazar fracciones y comandos de LaTeX
        processedLatex = this.replaceFractions(processedLatex);
        processedLatex = processedLatex.replace(/\\sqrt\{([^{}]+)\}/g, 'sqrt($1)');
        processedLatex = processedLatex.replace(/\\left|\\right/g, '');
        processedLatex = processedLatex.replace(/\\cdot|\\times/g, '*');
        processedLatex = processedLatex.replace(/\\div/g, '/');
        processedLatex = processedLatex.replace(/\\pi/g, 'pi');
        processedLatex = processedLatex.replace(/\\exp\{([^{}]+)\}/g, 'exp($1)');

        // Insertar '*' implícito
        processedLatex = processedLatex.replace(/(\d)\s*([a-zA-Z(])/g, '$1*$2');
        processedLatex = processedLatex.replace(/(\))\s*([a-zA-Z(])/g, '$1*$2');
        return processedLatex;
    }

    // Convertir la función g a JavaScript
    gFunctionToJavaScript(gFunc) {
        let processedGFunc = gFunc;
        console.log("Original gFunction LaTeX:", processedGFunc);

        // Corregir fracciones mal formadas
        processedGFunc = this.replaceFractions(processedGFunc);
        console.log("Después de reemplazar fracciones en gFunction:", processedGFunc);

        // Reemplazar otros comandos matemáticos
        processedGFunc = processedGFunc.replace(/\\sqrt\{([^{}]+)\}/g, 'sqrt($1)');
        processedGFunc = processedGFunc.replace(/\\left|\\right/g, '');
        processedGFunc = processedGFunc.replace(/\\cdot|\\times/g, '*');
        processedGFunc = processedGFunc.replace(/\\div/g, '/');
        processedGFunc = processedGFunc.replace(/\\pi/g, 'pi');
        processedGFunc = processedGFunc.replace(/\\ln/g, 'log');
        processedGFunc = processedGFunc.replace(/\\log/g, 'log10');
        processedGFunc = processedGFunc.replace(/\\exp\{([^{}]+)\}/g, 'exp($1)');
        processedGFunc = processedGFunc.replace(/\\sin/g, 'sin');
        processedGFunc = processedGFunc.replace(/\\cos/g, 'cos');
        processedGFunc = processedGFunc.replace(/\\tan/g, 'tan');

        // Reglas de multiplicación implícita
        processedGFunc = processedGFunc.replace(/(\d)([a-zA-Z(])/g, '$1*$2');
        processedGFunc = processedGFunc.replace(/([a-zA-Z)])([a-zA-Z(])/g, '$1*$2');
        processedGFunc = processedGFunc.replace(/\)([a-zA-Z(])/g, ')*$1');

        // Insertar '*' entre ')' y cualquier letra
        processedGFunc = processedGFunc.replace(/([a-zA-Z)])([a-zA-Z(])/g, '$1*$2');
        console.log("Después de insertar '*', gFunction:", processedGFunc);

        // Validar paréntesis balanceados
        const openParens = (processedGFunc.match(/\(/g) || []).length;
        const closeParens = (processedGFunc.match(/\)/g) || []).length;
        if (openParens !== closeParens) {
            throw new Error("gFunction contiene paréntesis desbalanceados.");
        }

        return processedGFunc;
    }

    static solveMethod(method, formData) {
        switch (method) {
            case 'jacobi':
                // Lógica del método Jacobi
                break;
            case 'gauss_seidel':
                // Lógica del método Gauss-Seidel
                break;
            case 'bisection':
                // Lógica del método Bisección
                break;
            case 'fixed_point':
                    // Lógica para resolver el método de punto fijo
                    break;
            default:
                throw new Error('Método desconocido');
        }
    }

    // Método para integrar la ecuación usando el método proporcionado
    integrate(method, func, limits) {
        switch (method) {
            case 'simpson':
                return this.simpson(func, limits);
            case 'trapezoidal':
                return this.trapezoidal(func, limits);
            default:
                throw new Error(`Método ${method} no soportado.`);
        }
    }

    // Método específico de integración
    simpson(func, limits) {
        // Implementación de Simpson
    }

    trapezoidal(func, limits) {
        // Implementación de Trapecio
    }
    async calculate(data) {
        try {
            const response = await fetch('/calculate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error(`Error del servidor: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Error al calcular:', error);
            throw error;
        }
    }
} 