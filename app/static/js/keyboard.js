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

// keyboard.js

// keyboard.js

// Global variable to track the currently active MathQuill field
let currentActiveMathField = null;

function initializeKeyboardFocus() {
    console.log("Inicializando el manejo del foco del teclado");
    
    // Comprehensive selector for all potential MathQuill and input fields
    const inputSelectors = [
        // Main equation input
        '#math-input',
        
        // Fixed Point method g(x) input
        '#gFunctionInput',
        
        // System of equations inputs
        '[id^="mathquill_equation_"]',
        
        // Other specific method inputs
        '#initial_guess',
        '#initial_guess_system',
        '#a_bisection',
        '#b_bisection',
        '#x0',
        '#x1',
        '#a_integration',
        '#b_integration'
    ];

    // Combine all selectors
    const allInputFields = document.querySelectorAll(inputSelectors.join(', '));

    allInputFields.forEach(field => {
        // Skip if already initialized
        if (field.mathquillInstance) return;

        // Initialize MathQuill for appropriate fields
        const mathquillEligibleFields = [
            '#math-input', 
            '#gFunctionInput', 
            '[id^="mathquill_equation_"]'
        ];

        if (mathquillEligibleFields.some(selector => 
            field.matches(selector))) {
            const MQ = MathQuill.getInterface(2);
            field.mathquillInstance = MQ.MathField(field, {
                spaceBehavesLikeTab: false,
                handlers: {
                    enter: function() {
                        const form = document.getElementById('calculator-form');
                        if (form) {
                            form.requestSubmit();
                        }
                    }
                }
            });
        }

        // Add event listeners for focus and click
        field.addEventListener('click', (e) => {
            e.preventDefault();
            
            // For MathQuill fields, use mathquillInstance
            if (field.mathquillInstance) {
                currentActiveMathField = field.mathquillInstance;
            } else {
                // For regular input fields, use the field directly
                currentActiveMathField = field;
            }
            
            console.log(`Campo activo actualizado a: ${field.id}`);
            field.focus();
        });

        field.addEventListener('focus', (e) => {
            // Similar logic as click event
            if (field.mathquillInstance) {
                currentActiveMathField = field.mathquillInstance;
            } else {
                currentActiveMathField = field;
            }
            
            console.log(`Foco establecido en campo: ${field.id}`);
        });
    });
}

function handleKeyPress(key) {
    console.log(`Manejando la tecla: ${key}`);

    // Intentar obtener el campo activo
    let currentField = currentActiveMathField;

    // Si no hay campo activo, buscar el campo enfocado
    if (!currentField) {
        const focusedField = document.activeElement;
        
        // Check if the focused field is one of our target fields
        const targetSelectors = [
            '.mathquill-field', 
            '#math-input', 
            '#gFunctionInput', 
            '[id^="mathquill_equation_"]',
            '#initial_guess',
            '#initial_guess_system',
            '#a_bisection',
            '#b_bisection',
            '#x0',
            '#x1',
            '#a_integration',
            '#b_integration'
        ];
        
        if (targetSelectors.some(selector => focusedField.matches(selector))) {
            currentField = focusedField.mathquillInstance || focusedField;
        }
    }

    // Si aún no se encuentra un campo, usar el primer campo disponible
    if (!currentField) {
        const firstField = document.querySelector([
            '.mathquill-field', 
            '#math-input', 
            '#gFunctionInput', 
            '[id^="mathquill_equation_"]'
        ].join(', '));
        
        if (firstField) {
            currentField = firstField.mathquillInstance || firstField;
        }
    }

    // Si después de todo no se encuentra un campo, mostrar advertencia
    if (!currentField) {
        console.warn("No hay un campo activo. Por favor, seleccione un campo para ingresar datos.");
        return;
    }

    // Manejar la inserción de símbolos de manera diferente para MathQuill y campos regulares
    if (currentField.latex && typeof currentField.latex === 'function') {
        // Es un campo MathQuill
        insertSymbol(currentField, key);
    } else {
        // Es un campo de entrada regular
        if (key === 'del') {
            currentField.value = currentField.value.slice(0, -1);
        } else if (key !== 'enter') {
            currentField.value += key;
        }
    }
}

// Mantener la función insertSymbol existente para campos MathQuill
// ... (el resto de tu código insertSymbol permanece igual)



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
// Inicializar cuando el DOM esté listo
document.addEventListener("DOMContentLoaded", () => {
    console.log("DOMContentLoaded en keyboard.js");
    loadKeyboard();
    initializeKeyboardFocus();

    // Re-inicializar cuando se agreguen nuevas ecuaciones dinámicamente
    const addEquationBtn = document.getElementById('addEquationBtn');
    if (addEquationBtn) {
        addEquationBtn.addEventListener('click', () => {
            setTimeout(initializeKeyboardFocus, 100);
        });
    }
});