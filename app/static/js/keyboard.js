// Inicialización de MathQuill
var MQ = MathQuill.getInterface(2);

// Definición del teclado
const keyboard_keys = [
    [
        {"key":"AC", "print_value": '../static/resources/imgs/ac.png'},
        {"key":"^2", "print_value": '\\Box^2'},
        {"key":"^{", "print_value": '^'}, // Puedes ajustar el print_value según prefieras
        {"key":"\\sqrt{}", "print_value": '\\sqrt{\\Box}'},
        {"key":"x", "print_value": 'x'},
        {"key":"7", "print_value": '7'},
        {"key":"8", "print_value": '8'},
        {"key":"9", "print_value": '9'},
        {"key":"+","print_value": '+'},
        {"key":"-","print_value": '-'},
    ],
    [
        {"key":"csc()", "print_value": 'csc'},
        {"key":"sin()", "print_value": 'sin'},
        {"key":"**-1", "print_value": '\\Box^{-1}'},
        {"key":"()/()", "print_value": '\\frac{\\Box}{\\Box}'}, // Fracción
        {"key":"y", "print_value": 'y'},
        {"key":"4", "print_value": '4'},
        {"key":"5", "print_value": '5'},
        {"key":"6", "print_value": '6'},
        {"key":"*","print_value": '\\times'},
        {"key":"/","print_value": '\\div'}
    ],
    [
        {"key":"sec()", "print_value": 'sec'},
        {"key":"cos()", "print_value": 'cos'},
        {"key":"ln()", "print_value": 'ln'},
        {"key":"log()", "print_value": 'log'},
        {"key":"e", "print_value": 'e'},
        {"key":"1", "print_value": '1'},
        {"key":"2", "print_value": '2'},
        {"key":"3", "print_value": '3'},
        {"key":"del", "print_value": '../static/resources/imgs/back.svg'}
    ],
    [
        {"key":"cot()", "print_value": 'cot'},
        {"key":"tan()", "print_value": 'tan'},
        {"key":"(", "print_value": '('},
        {"key":")", "print_value": ')'},
        {"key":"pi", "print_value": '\\pi'},
        {"key":"0", "print_value": '0'},
        {"key":".", "print_value": '.'},
        {"key":"right", "print_value": '../static/resources/imgs/right.svg'},
        {"key":"left", "print_value": '../static/resources/imgs/left.svg'},
        {"key":"enter", "print_value": '../static/resources/imgs/return.svg'}
    ]
];

const special_keys = ["del", "left", "right", "enter", "AC"];
const mathFieldSpan = document.getElementById('math-input');
let currentMathField = null;

document.addEventListener("DOMContentLoaded", () => {
    // Inicializar MathQuill en el campo principal
    currentMathField = MQ.MathField(mathFieldSpan, {
        spaceBehavesLikeTab: true,
        handlers: {
            edit: function() {
                // Actualizar el input hidden con el valor
                const latex = currentMathField.latex();
                document.getElementById('equation').value = latexToJavaScript(latex);
            }
        }
    });

    loadKeyboard();
});

function loadKeyboard() {
    const keyboard = document.getElementById("keyboard");
    keyboard.innerHTML = ''; // Limpiar el teclado existente

    keyboard_keys.forEach(rows => {
        let rowDiv = document.createElement('div');
        rowDiv.classList.add("row-keyboard");
        keyboard.appendChild(rowDiv);
        
        rows.forEach(key => {
            let button = document.createElement('button');
            button.type = 'button'; // Importante para evitar el envío del formulario
            button.classList.add("calc-btn");
            
            // Manejar teclas especiales
            if (special_keys.includes(key["key"])) {
                if (key["key"] === "del") {
                    button.classList.add("del-btn");
                }
                let img = document.createElement('img');
                img.src = key["print_value"];
                img.classList.add("btn-icon");
                button.appendChild(img);
            } else {
                // Renderizar símbolos matemáticos
                katex.render(key["print_value"], button, { 
                    throwOnError: false,
                    displayMode: false
                });
            }

            button.addEventListener('click', (e) => {
                e.preventDefault();
                handleKeyPress(key["key"]);
            });
            
            rowDiv.appendChild(button);
        });
    });
}

function handleKeyPress(key) {
    if (!currentMathField) return;

    // Obtener el latex actual
    const currentLatex = currentMathField.latex();

    // Lista de símbolos que deben evitar repeticiones
    const avoidRepeats = ['+', '-', 'x', '\\pi'];

    // Definir una función para evitar repeticiones
    const avoidRepeatingSigns = (newSign) => {
        const lastChar = currentLatex.slice(-1); // Obtener el último carácter
        return lastChar === newSign; // Verificar si es el mismo que el nuevo símbolo
    };

    const actions = {
        "AC": () => currentMathField.latex(""),
        "del": () => currentMathField.keystroke('Backspace'),
        "left": () => currentMathField.keystroke('Left'),
        "right": () => currentMathField.keystroke('Right'),
        "enter": () => document.getElementById('equationForm').requestSubmit(),
        "pi": () => {
            if (!avoidRepeatingSigns('\\pi')) {
                currentMathField.write("\\pi");
            }
        },
        "x": () => {
            if (!avoidRepeatingSigns('x')) {
                currentMathField.write('x');
            }
        },
        "+": () => {
            if (!avoidRepeatingSigns('+')) {
                currentMathField.write('+');
            }
        },
        "-": () => {
            if (!avoidRepeatingSigns('-')) {
                currentMathField.write('-');
            }
        },
        "^2": () => {
            currentMathField.write("^2");
        },
        "^{": () => {
            currentMathField.write("^{}");
            currentMathField.keystroke('Left'); // Mueve el cursor dentro de las llaves
        },
        "()/()": () => { // Manejar la fracción
            currentMathField.write("\\frac{}{}");
            currentMathField.keystroke('Left'); // Coloca el cursor en el numerador
            currentMathField.keystroke('Left'); // Opcional: ajusta la posición del cursor
        },
        // Puedes añadir más acciones aquí si es necesario
    };

    // Verificar si la tecla tiene una acción definida
    if (actions[key]) {
        actions[key]();
    } else if (key.endsWith("()")) {
        currentMathField.write(key.slice(0, -2) + "\\left(\\right)");
        currentMathField.keystroke('Left');
    } else {
        // Para otros símbolos, verificar repetición
        if (!avoidRepeatingSigns(key)) {
            currentMathField.write(key);
        }
    }

    currentMathField.focus();
}

// Función auxiliar para convertir LaTeX a JavaScript
function latexToJavaScript(latex) {
    return latex
        .replace(/\\cdot/g, '*')
        .replace(/\\frac{([^}]*)}{([^}]*)}/g, '($1)/($2)')
        .replace(/\^/g, '**')
        .replace(/\\pi/g, 'Math.PI')
        .replace(/\\sin/g, 'Math.sin')
        .replace(/\\cos/g, 'Math.cos')
        .replace(/\\tan/g, 'Math.tan')
        .replace(/\\csc/g, '1/Math.sin')
        .replace(/\\sec/g, '1/Math.cos')
        .replace(/\\cot/g, '1/Math.tan')
        .replace(/\\ln/g, 'Math.log')
        .replace(/\\log/g, 'Math.log10');
}
