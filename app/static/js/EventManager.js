class EventManager {
    constructor(uiManager, mathFieldManager, formHandler) {
        this.uiManager = uiManager;
        this.mathFieldManager = mathFieldManager;
        this.formHandler = formHandler;
    }

    setupEvents() {
        const form = this.uiManager.getElement('calculator-form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.formHandler.handleSubmit();
        });

        const methodSelect = this.uiManager.getElement('method');
        methodSelect.addEventListener('change', (e) => {
            this.formHandler.handleMethodChange(e.target.value);
        });
    }
}