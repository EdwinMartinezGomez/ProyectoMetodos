class EquationManager {
    constructor() {
        this.equation = null;
    }
    static validateEquation(equation) {
        // Validar la ecuación proporcionada
        if (!equation) {
            throw new Error('La ecuación no es válida');
        }
    }

    static parseEquations(equations) {
        // Parsear las ecuaciones
        return equations.split(';').map(eq => eq.trim());
    }
    setEquation(func) {
        this.equation = func;
    }

    getEquation() {
        return this.equation;
    }
    static validateEquation(equation) {
        // Validación de la ecuación
    }

    static parseEquations(equations) {
        // Parsear las ecuaciones
    }
}
