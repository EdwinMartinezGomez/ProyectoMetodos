import {CalculatorApp} from './CalculatorApp.js'; 

// Inicializar la aplicación al cargar el DOM
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.calculatorApp = new CalculatorApp();
        console.log("CalculatorApp inicializado y asignado a window.calculatorApp.");
    } catch (error) {
        console.error('Error fatal al inicializar la aplicación:', error);
        document.body.innerHTML = `
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error Fatal</h4>
                    <p>No se pudo iniciar la aplicación. Por favor, recargue la página o contacte al soporte técnico.</p>
                    <button class="btn btn-primary mt-3" onclick="location.reload()">Recargar Página</button>
                </div>
            </div>
        `;
    }
});
