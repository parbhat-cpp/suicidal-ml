let BASE_URL = window.location.href;
BASE_URL = BASE_URL[BASE_URL.length - 1] === '/' ? BASE_URL.substring(0, BASE_URL.length - 1) : BASE_URL;

const predictionOutput = document.getElementById('prediction-output');
const predictBtn = document.getElementById('predict-btn');

predictBtn.addEventListener('click', async function () {
    const userTextInput = document.getElementById('text').value.trim();

    if (!userTextInput) {
        return;
    }

    predictionOutput.innerHTML = `
        <div class="loader"></div>
    `;

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

        const prediction = predictionData['prediction'];

        if (prediction === 0.0) {
            predictionOutput.innerHTML = `
                <div class="prediction">
                    <h1>
                        No signs of suicidal ideation detected.
                    </h1>
                    <p>
                        This is an AI-based prediction. If you're in distress, please reach out to a helpline or
                        professional support.
                    </p>
                </div>
            `;
        } else {
            predictionOutput.innerHTML = `
                <div class="prediction">
                    <h1>
                        Warning: Signs of suicidal ideation detected in the text
                    </h1>
                    <p>
                        This is an AI-based prediction. If you're in distress, please reach out to a helpline or
                        professional support.
                    </p>
                </div>
            `;
        }
    } else {
        alert('Prediction failed! Please try again');
        predictionOutput.innerHTML = `
            <div class="prediction">
                <h1>🔍 Awaiting Input...</h1>
                <p>Type something you're feeling, and we'll analyze it for signs of distress.</p>
            </div>
        `;
    }

    document.getElementById('text').value = '';
});
