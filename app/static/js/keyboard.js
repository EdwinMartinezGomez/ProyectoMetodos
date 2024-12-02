// keyboard.js

// Definición del teclado
const keyboard_keys = [
    [
        { "key": "AC", "print_value": '../static/resources/imgs/ac.png' },
        { "key": "^2", "print_value": '\\Box^2' },
        { "key": "^{", "print_value": '^' }, // Puedes ajustar el print_value según prefieras
        { "key": "\\sqrt{}", "print_value": '\\sqrt{\\Box}' },
        { "key": "x", "print_value": 'x' },
        { "key": "7", "print_value": '7' },
        { "key": "8", "print_value": '8' },
        { "key": "9", "print_value": '9' },
        { "key": "+", "print_value": '+' },
        { "key": "-", "print_value": '-' },
    ],
    [
        { "key": "csc()", "print_value": 'csc' },
        { "key": "sin()", "print_value": 'sin' },
        { "key": "**-1", "print_value": '\\Box^{-1}' },
        { "key": "()/()", "print_value": '\\frac{\\Box}{\\Box}' }, // Fracción
        { "key": "y", "print_value": 'y' },
        { "key": "4", "print_value": '4' },
        { "key": "5", "print_value": '5' },
        { "key": "6", "print_value": '6' },
        { "key": "*", "print_value": '\\times' },
        { "key": "/", "print_value": '\\div' }
    ],
    [
        { "key": "sec()", "print_value": 'sec' },
        { "key": "cos()", "print_value": 'cos' },
        { "key": "ln()", "print_value": 'ln' },
        { "key": "log()", "print_value": 'log' },
        { "key": "e", "print_value": 'e' },
        { "key": "1", "print_value": '1' },
        { "key": "2", "print_value": '2' },
        { "key": "3", "print_value": '3' },
        { "key": "del", "print_value": '../static/resources/imgs/back.svg' }
    ],
    [
        { "key": "cot()", "print_value": 'cot' },
        { "key": "tan()", "print_value": 'tan' },
        { "key": "(", "print_value": '(' },
        { "key": ")", "print_value": ')' },
        { "key": "pi", "print_value": '\\pi' },
        { "key": "0", "print_value": '0' },
        { "key": ".", "print_value": '.' },
        { "key": "right", "print_value": '../static/resources/imgs/right.svg' },
        { "key": "left", "print_value": '../static/resources/imgs/left.svg' },
        { "key": "enter", "print_value": '../static/resources/imgs/return.svg' }
    ]
];

const special_keys = ["del", "left", "right", "enter", "AC"];

document.addEventListener("DOMContentLoaded", () => {
    console.log("DOMContentLoaded en keyboard.js");
    loadKeyboard();
    initializeKeyboardFocus();
});
// Nueva función para inicializar el manejo del foco
function initializeKeyboardFocus() {
    // Obtener todos los campos MathQuill
    const mathquillFields = document.querySelectorAll('.mathquill-field, #math-input');

    mathquillFields.forEach(field => {
        // Cuando se hace clic en un campo, actualizar el campo activo
        field.addEventListener('click', (e) => {
            e.preventDefault();
            if (window.calculatorApp) {
                // Buscar el identificador del campo
                const fieldId = field.id;
                const mathField = window.calculatorApp.getMathFieldByElement(field);
                if (mathField) {
                    window.calculatorApp.setActiveMathField(mathField);
                    console.log(`Campo activo actualizado a: ${fieldId}`);
                }
            }
        });

        // Manejar el foco
        field.addEventListener('focus', (e) => {
            if (window.calculatorApp) {
                const mathField = window.calculatorApp.getMathFieldByElement(field);
                if (mathField) {
                    window.calculatorApp.setActiveMathField(mathField);
                    console.log(`Foco establecido en campo: ${field.id}`);
                }
            }
        });
    });
}
class Keyboard {
    constructor() {
        this.keyboardElement = document.getElementById('keyboard');
    }

    // Método para mostrar u ocultar el teclado
    toggleVisibility() {
        this.keyboardElement.classList.toggle('hidden');
        this.keyboardElement = document.getElementById('keyboard');
    }
    show() {
        this.keyboardElement.style.display = 'block';
    }

    hide() {
        this.keyboardElement.style.display = 'none';
    }

    // Método para presionar una tecla
    pressKey(key) {
        // Enviar la tecla presionada al campo de entrada
    }
}


function loadKeyboard() {
    console.log("Cargando el teclado virtual...");
    const keyboard = document.getElementById("keyboard");
    if (!keyboard) {
        console.error("No se encontró el contenedor del teclado (id='keyboard').");
        return;
    }
    keyboard.innerHTML = '';

    keyboard_keys.forEach(rows => {
        let rowDiv = document.createElement('div');
        rowDiv.classList.add("row-keyboard");
        keyboard.appendChild(rowDiv);

        rows.forEach(key => {
            let button = document.createElement('button');
            button.type = 'button';
            button.classList.add("calc-btn");
            button.setAttribute('tabindex', '-1');
            button.setAttribute('aria-label', key["key"]);

            if (special_keys.includes(key["key"])) {
                if (key["key"] === "del") {
                    button.classList.add("del-btn");
                }
                let img = document.createElement('img');
                img.src = key["print_value"];
                img.classList.add("btn-icon");
                button.appendChild(img);
            } else {
                try {
                    katex.render(key["print_value"], button, {
                        throwOnError: false,
                        displayMode: false
                    });
                } catch (error) {
                    console.error(`Error al renderizar el símbolo: ${key["print_value"]}`, error);
                }
            }

            button.addEventListener('mousedown', (e) => {
                e.preventDefault();
            });

            button.addEventListener('click', (e) => {
                e.preventDefault();
                console.log(`Botón presionado: ${key["key"]}`);
                handleKeyPress(key["key"]);
            });

            button.addEventListener('mouseup', (e) => {
                e.preventDefault();
            });

            rowDiv.appendChild(button);
        });
    });
    console.log("Teclado virtual cargado correctamente.");
}
function handleKeyPress(key) {
    console.log(`Manejando la tecla: ${key}`);

    // Verificar si CalculatorApp está inicializado y hay un campo activo
    if (!window.calculatorApp) {
        console.warn("CalculatorApp no está inicializado.");
        return;
    }

    let currentMathField = window.calculatorApp.getActiveMathField();

    if (!currentMathField) {
        console.warn("No hay un campo MathQuill activo. Por favor, seleccione un campo para ingresar datos.");
        return;
    }

    insertSymbol(currentMathField, key);
}


// Función para insertar el símbolo en el campo MathQuill activo
function insertSymbol(currentMathField, key) {
    console.log("mathField activo:", currentMathField);
    // Obtener el LaTeX actual
    const currentLatex = currentMathField.latex();
    console.log(`LaTeX actual: ${currentLatex}`);

    // Lista de símbolos que deben evitar repeticiones
    const avoidRepeats = ['+', '-', 'x', '\\pi'];

    // Definir una función para evitar repeticiones
    const avoidRepeatingSigns = (newSign) => {
        const lastChar = currentLatex.slice(-1); // Obtener el último carácter
        return avoidRepeats.includes(newSign) && lastChar === newSign;
    };

    const actions = {
        "AC": () => {
            console.log("Acción: AC");
            currentMathField.latex("");
        },
        "del": () => {
            console.log("Acción: del");
            currentMathField.keystroke('Backspace');
        },
        "left": () => {
            console.log("Acción: left");
            currentMathField.keystroke('Left');
        },
        "right": () => {
            console.log("Acción: right");
            currentMathField.keystroke('Right');
        },
        "enter": () => {
            console.log("Acción: enter");
            const form = document.getElementById('calculator-form');
            if (form) {
                form.requestSubmit();
            } else {
                console.error("No se encontró el formulario (id='calculator-form').");
            }
        },
        "pi": () => {
            console.log("Acción: pi");
            if (!avoidRepeatingSigns('\\pi')) {
                currentMathField.write("\\pi");
            }
        },
        "x": () => {
            console.log("Acción: x");
            if (!avoidRepeatingSigns('x')) {
                currentMathField.write('x');
            }
        },
        "+": () => {
            console.log("Acción: +");
            if (!avoidRepeatingSigns('+')) {
                currentMathField.write('+');
            }
        },
        "-": () => {
            console.log("Acción: -");
            if (!avoidRepeatingSigns('-')) {
                currentMathField.write('-');
            }
        },
        "^2": () => {
            console.log("Acción: ^2");
            currentMathField.write("^2");
        },
        "^{": () => {
            console.log("Acción: ^{");
            currentMathField.write("^{}");
            currentMathField.keystroke('Left'); // Mueve el cursor dentro de las llaves
        },
        "()/()": () => { // Manejar la fracción
            console.log("Acción: fracción");
            currentMathField.write("\\frac{}{}");
            currentMathField.keystroke('Left'); // Coloca el cursor en el numerador
            currentMathField.keystroke('Left'); // Opcional: ajusta la posición del cursor
        },
        "csc()": () => {
            console.log("Acción: csc()");
            currentMathField.write("\\csc()");
            currentMathField.keystroke('Left'); // Coloca el cursor dentro de los paréntesis
        },
        "sec()": () => {
            console.log("Acción: sec()");
            currentMathField.write("\\sec()");
            currentMathField.keystroke('Left');
        },
        "cot()": () => {
            console.log("Acción: cot()");
            currentMathField.write("\\cot()");
            currentMathField.keystroke('Left');
        },
        "sin()": () => {
            console.log("Acción: sin()");
            currentMathField.write("\\sin()");
            currentMathField.keystroke('Left');
        },
        "cos()": () => {
            console.log("Acción: cos()");
            currentMathField.write("\\cos()");
            currentMathField.keystroke('Left');
        },
        "tan()": () => {
            console.log("Acción: tan()");
            currentMathField.write("\\tan()");
            currentMathField.keystroke('Left');
        },
        "ln()": () => {
            console.log("Acción: ln()");
            currentMathField.write("\\ln()");
            currentMathField.keystroke('Left');
        },
        "log()": () => {
            console.log("Acción: log()");
            currentMathField.write("\\log()");
            currentMathField.keystroke('Left');
        },
        // Puedes añadir más acciones aquí si es necesario
    };

    // Verificar si la tecla tiene una acción definida
    if (actions[key]) {
        actions[key]();
    } else if (key.endsWith("()")) {
        console.log(`Acción: insertar paréntesis para ${key}`);
        currentMathField.write(key.slice(0, -2) + "\\left(\\right)");
        currentMathField.keystroke('Left');
    } else {
        // Para otros símbolos, verificar repetición
        console.log(`Acción: insertar símbolo ${key}`);
        if (!avoidRepeatingSigns(key)) {
            currentMathField.write(key);
        }
    }

    // **Re-enfocar el campo de MathQuill después de manejar la tecla**
    currentMathField.focus();
    console.log("Campo MathQuill re-enfocado.");
}
