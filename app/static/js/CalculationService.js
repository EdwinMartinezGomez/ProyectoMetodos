class CalculationService {
    async calculate(data) {
        try {
            const response = await fetch('/calculate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error(`Error del servidor: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Error al calcular:', error);
            throw error;
        }
    }
} 