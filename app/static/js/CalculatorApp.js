class CalculatorApp {
    constructor() {
        try {
            this.uiManager = new UIManager();
            this.uiManager.initializeElements([
                'calculator-form', 'math-input', 'method', 'equation',
                'results', 'resultTable'
            ]);
            this.mathFieldManager = new MathFieldManager(this.uiManager);
            this.eventManager = new EventManager(this.uiManager, this.mathFieldManager);
            this.formHandler = new FormHandler(this.uiManager);
            this.calculationService = new CalculationService();
            this.resultRenderer = new ResultRenderer(this.uiManager);

            this.eventManager.setupGlobalListeners(
                this.formHandler,
                this.calculationService,
                this.resultRenderer
            );

            console.log("CalculatorApp inicializado correctamente.");
        } catch (error) {
            this.handleInitializationError(error);
        }
    }

    handleInitializationError(error) {
        console.error('Error de inicialización:', error);
        document.body.innerHTML = `
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error de Inicialización</h4>
                    <p>No se pudo inicializar la calculadora.</p>
                    <hr>
                    <p class="mb-0">Error: ${error.message}</p>
                    <button class="btn btn-primary mt-3" onclick="location.reload()">Recargar Página</button>
                </div>
            </div>
        `;
    }
}
document.addEventListener('DOMContentLoaded', () => {
    new CalculatorApp();
});