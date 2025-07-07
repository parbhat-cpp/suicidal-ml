const BASE_URL = window.location.href;

const predictionOutput = document.getElementById('prediction-output');
const predictBtn = document.getElementById('predict-btn');

predictBtn.addEventListener('click', async function () {
    const userTextInput = document.getElementById('text').value;

    if (!userTextInput) {
        return;
    }

    const predictionResponse = await fetch(`${BASE_URL}/predict-mental-health`,
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                'text': userTextInput,
            }),
        }
    );

    if (predictionResponse.ok) {
        const predictionData = await predictionResponse.json();

        alert(predictionData);
    } else {
        alert('failed');
    }
});
