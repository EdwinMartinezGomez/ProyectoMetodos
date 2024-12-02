class InputValidator {
    static validateNumber(input) {
        return !isNaN(input) && input !== '';
    }

    static validateRange(min, max, value) {
        return value >= min && value <= max;
    }
    static validateNumber(value, name) {
        if (isNaN(value)) {
            throw new Error(`El valor de ${name} debe ser un número válido`);
        }
    }

    static validateRange(value, min, max, name) {
        if (value < min || value > max) {
            throw new Error(`${name} debe estar entre ${min} y ${max}`);
        }
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
}
