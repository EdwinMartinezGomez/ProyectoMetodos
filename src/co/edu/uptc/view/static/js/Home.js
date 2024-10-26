const metodoSelect = document.getElementById('metodo');
const funcionInput = document.getElementById('funcion');
const camposAdicionales = document.getElementById('camposAdicionales');
const form = document.getElementById('metodoForm');
const errorDiv = document.getElementById('error');
const teclado = document.querySelector('.teclado');
let chart;

metodoSelect.addEventListener('change', actualizarCampos);
form.addEventListener('submit', manejarEnvio);
funcionInput.addEventListener('input', actualizarGrafica);
teclado.addEventListener('click', manejarClickTeclado);
funcionInput.addEventListener('keydown', manejarTeclaEspecial);

function actualizarCampos() {
    camposAdicionales.innerHTML = '';
    const metodo = metodoSelect.value;
    if (metodo === 'biseccion' || metodo === 'secante') {
        agregarCampo('a', 'Valor de a:');
        agregarCampo('b', 'Valor de b:');
    }
    if (metodo === 'puntoFijo' || metodo === 'newtonRaphson') {
        agregarCampo('x0', 'Valor inicial x0:');
    }
    if (metodo === 'secante') {
        agregarCampo('x1', 'Valor inicial x1:');
    }
}

function agregarCampo(id, label) {
    const campo = `
        <label for="${id}">${label}</label>
        <input type="number" id="${id}" step="any" required>
    `;
    camposAdicionales.insertAdjacentHTML('beforeend', campo);
}

function manejarEnvio(e) {
    e.preventDefault();
    errorDiv.textContent = '';
    if (validarEntrada()) {
        const datos = recopilarDatos();
        console.log('Datos válidos:', datos);
        // Aquí se enviarían los datos al backend en Python
    }
}

function validarEntrada() {
    const metodo = metodoSelect.value;
    if (!metodo) {
        mostrarError('Por favor, seleccione un método numérico.');
        return false;
    }
    if (!funcionInput.value) {
        mostrarError('Por favor, ingrese una función.');
        return false;
    }
    return true;
}

function recopilarDatos() {
    const datos = {
        metodo: metodoSelect.value,
        funcion: funcionInput.value
    };
    camposAdicionales.querySelectorAll('input').forEach(input => {
        datos[input.id] = input.value;
    });
    return datos;
}

function mostrarError(mensaje) {
    errorDiv.textContent = mensaje;
}

function actualizarGrafica() {
    const funcion = funcionInput.value;
    if (!funcion) return;

    try {
        const xValues = [];
        const yValues = [];
        for (let x = -10; x <= 10; x += 0.5) {
            xValues.push(x);
            const funcionPython = funcion.replace(/np\./g, '');
            yValues.push(math.evaluate(funcionPython, { x: x }));
        }

        if (chart) {
            chart.destroy();
        }

        const ctx = document.getElementById('myChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: xValues,
                datasets: [{
                    label: 'f(x)',
                    data: yValues,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom'
                    }
                }
            }
        });
    } catch (err) {
        mostrarError('Error al evaluar la función. Por favor, verifique la sintaxis.');
    }
}

function manejarClickTeclado(e) {
    if (e.target.tagName === 'BUTTON') {
        const func = e.target.dataset.func;
        const cursorPos = funcionInput.selectionStart;
        const textoAnterior = funcionInput.value.substring(0, cursorPos);
        const textoPosterior = funcionInput.value.substring(cursorPos);
        funcionInput.value = textoAnterior + func + textoPosterior;
        funcionInput.focus();
        funcionInput.setSelectionRange(cursorPos + func.length, cursorPos + func.length);
        actualizarGrafica();
    }
}

function manejarTeclaEspecial(e) {
    if (e.altKey && e.keyCode === 94) { // Alt + 94 (^)
        e.preventDefault();
        const cursorPos = funcionInput.selectionStart;
        const textoAnterior = funcionInput.value.substring(0, cursorPos);
        const textoPosterior = funcionInput.value.substring(cursorPos);
        const exponente = prompt('Ingrese el exponente:');
        if (exponente !== null) {
            funcionInput.value = textoAnterior + '^(' + exponente + ')' + textoPosterior;
            actualizarGrafica();
        }
    }
}

// Inicializar la gráfica vacía al cargar la página
window.addEventListener('load', () => {
    const ctx = document.getElementById('myChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'f(x)',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            }
        }
    });
});