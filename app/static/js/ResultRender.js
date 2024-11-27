class ResultRenderer {
    constructor(uiManager) {
        this.uiManager = uiManager;
    }

    render(result) {
        const resultTable = this.uiManager.elements['resultTable'];
        resultTable.innerHTML = '';

        if (result.solution) {
            for (const [key, value] of Object.entries(result.solution)) {
                const row = document.createElement('tr');
                row.innerHTML = `<td>${key}</td><td>${value}</td>`;
                resultTable.appendChild(row);
            }
        }
    }
}