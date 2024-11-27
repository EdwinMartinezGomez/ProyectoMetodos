class UIManager {
    constructor() {
        this.elements = this.initializeElements();
    }

    initializeElements(ids) {
        ids.forEach(id => {
            const element = document.getElementById(id);
            if (!element) throw new Error(`Elemento con ID ${id} no encontrado`);
            this.elements[id] = element;
        });
    }

    getElement(id) {
        return this.elements[id];
    }
}