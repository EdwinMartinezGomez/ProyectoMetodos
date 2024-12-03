// Definición del teclado
const keyboard_keys = [
    [
        { "key": "AC", "print_value": '../static/resources/imgs/ac.png' },
        { "key": "^2", "print_value": '\\Box^2' },
        { "key": "^{", "print_value": '^' },
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
        { "key": "()/()", "print_value": '\\frac{\\Box}{\\Box}' },
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

// Variable global para rastrear el campo MathQuill activo
let currentActiveMathField = null;

// Inicializar cuando el DOM esté listo
document.addEventListener("DOMContentLoaded", () => {
    console.log("DOMContentLoaded en keyboard.js");
    loadKeyboard();
    initializeKeyboardFocus();

    // Escuchar el evento personalizado para re-inicializar campos MathQuill
    document.addEventListener('mathFieldsUpdated', () => {
        initializeKeyboardFocus();
    });
});

// Función para cargar el teclado virtual
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

            // Evitar el enfoque al presionar el botón
            button.addEventListener('mousedown', (e) => {
                e.preventDefault();
            });

            // Manejar el clic en el botón
            button.addEventListener('click', (e) => {
                e.preventDefault();
                console.log(`Botón presionado: ${key["key"]}`);
                handleKeyPress(key["key"]);
            });

            // Evitar comportamientos predeterminados al soltar el botón
            button.addEventListener('mouseup', (e) => {
                e.preventDefault();
            });

            rowDiv.appendChild(button);
        });
    });
    console.log("Teclado virtual cargado correctamente.");
}

const special_keys = ["del", "left", "right", "enter", "AC"];

// Función para inicializar el manejo del foco y los campos MathQuill
function initializeKeyboardFocus() {
    console.log("Inicializando el manejo del foco del teclado");

    // Selectores de todos los campos MathQuill y de entrada
    const inputSelectors = [
        '#math-input', // Prioridad alta
        '#gFunctionInput',
        '[id^="mathquill_equation_"]',
        '#initial_guess',
        '#initial_guess_system',
        '#a_bisection',
        '#b_bisection',
        '#x0',
        '#x1',
        '#a_integration',
        '#b_integration',
        '#n_integration'
    ];

    const allInputFields = document.querySelectorAll(inputSelectors.join(', '));

    allInputFields.forEach(field => {
        // Evitar re-inicialización
        if (field.mathquillInstance) return;

        // Verificar si el campo es un div MathQuill o un input regular
        const isMathQuillField = field.classList.contains('mathquill-field') || field.id === 'math-input' || field.id === 'gFunctionInput';

        if (isMathQuillField) {
            // Asumimos que las instancias de MathQuill ya están inicializadas en main.js
            // Solo añadimos los event listeners necesarios para actualizar currentActiveMathField

            // Obtener la instancia de MathQuill
            let mathFieldInstance = null;
            if (field.id === 'math-input' || field.id === 'gFunctionInput') {
                // Campos estáticos
                mathFieldInstance = window.calculatorApp.allMathFields.get(field.id);
            } else if (field.id.startsWith('mathquill_equation_')) {
                const equationNumber = field.id.split('_').pop();
                mathFieldInstance = window.calculatorApp.allMathFields.get(`equation_${equationNumber}`);
            }

            if (mathFieldInstance) {
                // Añadir event listeners para 'focus' y 'blur'
                const mathQuillContainer = mathFieldInstance.__container;
                if (mathQuillContainer) {
                    mathQuillContainer.addEventListener('focus', () => {
                        currentActiveMathField = mathFieldInstance;
                        console.log(`Campo MathQuill enfocado: ${mathQuillContainer.id}`);
                    });

                    mathQuillContainer.addEventListener('blur', () => {
                        console.log(`Campo MathQuill desenfocado: ${mathQuillContainer.id}`);
                    });
                }
            }
        } else {
            // Campos de entrada regulares (como initial_guess)
            field.addEventListener('focus', () => {
                currentActiveMathField = field;
                console.log(`Campo activo actualizado a: ${field.id}`);
            });
        }
    });

    // Actualizar el campo activo si alguno está enfocado actualmente
    const focusedElement = document.activeElement;
    if (focusedElement) {
        const isMathQuillField = focusedElement.classList.contains('mathquill-field') || focusedElement.id === 'math-input' || focusedElement.id === 'gFunctionInput';
        if (isMathQuillField) {
            if (focusedElement.id === 'math-input' || focusedElement.id === 'gFunctionInput') {
                currentActiveMathField = window.calculatorApp.allMathFields.get(focusedElement.id);
            } else if (focusedElement.id.startsWith('mathquill_equation_')) {
                const equationNumber = focusedElement.id.split('_').pop();
                currentActiveMathField = window.calculatorApp.allMathFields.get(`equation_${equationNumber}`);
            }
            console.log(`Campo activo inicializado a: ${focusedElement.id}`);
        } else {
            currentActiveMathField = focusedElement;
            console.log(`Campo activo inicializado a: ${focusedElement.id}`);
        }
    }
}

// Función para manejar la pulsación de teclas
function handleKeyPress(key) {
    console.log(`Manejando la tecla: ${key}`);

    // Intentar obtener el campo activo
    let currentField = currentActiveMathField;

    // Si no hay campo activo, buscar el campo enfocado
    if (!currentField) {
        const focusedField = document.activeElement;

        // Verificar si el campo enfocado es uno de los campos objetivo
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
            '#b_integration',
            '#n_integration'
        ];

        if (targetSelectors.some(selector => focusedField.matches(selector))) {
            if (focusedField.classList.contains('mathquill-field') || focusedField.id === 'math-input' || focusedField.id === 'gFunctionInput') {
                if (focusedField.id === 'math-input' || focusedField.id === 'gFunctionInput') {
                    currentField = window.calculatorApp.allMathFields.get(focusedField.id);
                } else if (focusedField.id.startsWith('mathquill_equation_')) {
                    const equationNumber = focusedField.id.split('_').pop();
                    currentField = window.calculatorApp.allMathFields.get(`equation_${equationNumber}`);
                }
            } else {
                currentField = focusedField;
            }
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
        } else if (key === 'enter') {
            const form = document.getElementById('calculator-form');
            if (form) {
                form.requestSubmit();
            }
        } else {
            currentField.value += key;
        }
    }
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
    if (currentMathField.focus) {
        currentMathField.focus();
    } else if (currentMathField.focused) {
        currentMathField.focused();
    }
    console.log("Campo MathQuill re-enfocado.");

    
}

// Mapeo de campos MathQuill a sus campos ocultos (solo para campos estáticos)
const hiddenInputMap = {
    'math-input': 'equation',
    'gFunctionInput': 'gFunctionHidden',
    // Los campos dinámicos seguirán una convención de nombres y no se incluyen aquí
};

// Función para sincronizar todos los campos MathQuill con sus campos ocultos
function synchronizeMathFields() {
    console.log("Sincronizando campos MathQuill con campos ocultos...");
    const mathFields = document.querySelectorAll('.mathquill-field, #math-input, #gFunctionInput');
    mathFields.forEach(field => {
        if (field.mathquillInstance) {
            let hiddenInputId;
            const fieldId = field.id;

            if (hiddenInputMap[fieldId]) {
                hiddenInputId = hiddenInputMap[fieldId];
            } else if (fieldId.startsWith('mathquill_equation_')) {
                const equationNumber = fieldId.split('_').pop();
                hiddenInputId = `equation_${equationNumber}`;
            }

            if (hiddenInputId) {
                const hiddenInput = document.getElementById(hiddenInputId);
                if (hiddenInput) {
                    hiddenInput.value = field.mathquillInstance.latex();
                    console.log(`Campo oculto sincronizado: ${hiddenInput.id} = ${hiddenInput.value}`);
                } else {
                    console.warn(`No se encontró el campo oculto para: ${fieldId}`);
                }
            } else {
                console.warn(`No hay mapeo para el campo: ${fieldId}`);
            }
        }
    });
}

// Función para manejar la sincronización inicial mediante un observer
const observer = new MutationObserver((mutationsList, observer) => {
    for (let mutation of mutationsList) {
        if (mutation.type === 'childList') {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    const mathQuillFields = node.querySelectorAll('.mathquill-field, #math-input, #gFunctionInput');
                    mathQuillFields.forEach(field => {
                        if (!field.mathquillInstance) {
                            // Aquí deberías inicializar MathQuill si aún no lo está
                            // Sin embargo, en este diseño, main.js se encarga de inicializar MathQuill
                            // Por lo tanto, solo necesitas asegurarte de que los event listeners estén actualizados
                            initializeKeyboardFocus();
                        }
                    });
                }
            });
        }
    }
});

// Configurar el observer para el contenedor de ecuaciones y variables
const equationsContainer = document.getElementById('equationsContainer');
if (equationsContainer) {
    observer.observe(equationsContainer, { childList: true, subtree: true });
}
