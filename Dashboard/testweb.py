import time
import board
import adafruit_bme680
from flask import Flask, jsonify, render_template, request
import csv
from datetime import datetime

# Set up the BME688 sensor
i2c = board.I2C()  # SCL=GPIO3, SDA=GPIO2
bme = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x77)
bme.sea_level_pressure = 1013.25  # optional, adjust if you know local pressure

# Set up Flask
app = Flask(__name__)

# Store data for export
data_log = []

@app.route("/")
def index():
    """
    Serve the main HTML page.
    """
    return """
<!DOCTYPE html>
<html>
<head>
    <title>BME688 Sensor Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>BME688 Sensor Dashboard</h1>
    
    <div>
        <p>Temperature: <span id="temp"></span> °C</p>
        <p>Humidity: <span id="humid"></span> %</p>
        <p>Pressure: <span id="press"></span> hPa</p>
        <p>Gas: <span id="gas"></span> ohms</p>
    </div>
    
    <canvas id="myChart" width="400" height="200"></canvas>
    
    <button onclick="exportData()">Export Data</button>
    
    <script>
        const labels = [];
        const tempData = [];
        const humidData = [];
        const gasData = [];

        const ctx = document.getElementById('myChart').getContext('2d');
        const myChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Temperature (°C)',
                        data: tempData,
                        borderColor: 'rgb(255, 99, 132)',
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Humidity (%)',
                        data: humidData,
                        borderColor: 'rgb(54, 162, 235)',
                        fill: false,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Gas (Ohms)',
                        data: gasData,
                        borderColor: 'rgb(255, 205, 86)',
                        fill: false,
                        yAxisID: 'y2'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Temperature (°C)'
                        },
                        suggestedMin: 0
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Humidity (%)'
                        },
                        grid: {
                            drawOnChartArea: false
                        },
                        suggestedMin: 0
                    },
                    y2: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Gas (Ohms)'
                        },
                        grid: {
                            drawOnChartArea: false
                        },
                        suggestedMin: 0
                    }
                }
            }
        });

        async function fetchData() {
            const response = await fetch('/sensor');
            const data = await response.json();

            // Update text fields
            document.getElementById('temp').textContent = data.temperature.toFixed(2);
            document.getElementById('humid').textContent = data.humidity.toFixed(2);
            document.getElementById('press').textContent = data.pressure.toFixed(2);
            document.getElementById('gas').textContent = data.gas.toFixed(2);

            // Add data to chart
            const now = new Date().toLocaleTimeString();
            labels.push(now);
            tempData.push(data.temperature);
            humidData.push(data.humidity);
            gasData.push(data.gas);

            if (labels.length > 30) {
                labels.shift();
                tempData.shift();
                humidData.shift();
                gasData.shift();
            }

            myChart.update();
        }

        // Export Data
        async function exportData() {
            const response = await fetch('/export');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sensor_data.csv';
            a.click();
            window.URL.revokeObjectURL(url);
        }

        // Poll the sensor every 2 seconds
        setInterval(fetchData, 2000);
    </script>
</body>
</html>
"""

@app.route("/sensor")
def sensor_data():
    """
    Returns the current sensor readings as JSON.
    """
    temperature = bme.temperature
    humidity = bme.humidity
    pressure = bme.pressure
    gas = bme.gas
    
    # Log data for export
    data_log.append({
        "timestamp": datetime.now().isoformat(),
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "gas": gas
    })
    
    # Limit log size to avoid memory issues
    if len(data_log) > 1000:
        data_log.pop(0)
    
    return jsonify({
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "gas": gas
    })

@app.route("/export")
def export_data():
    """
    Export data log as CSV.
    """
    def generate():
        data = data_log
        header = "timestamp,temperature,humidity,pressure,gas\n"
        yield header
        for entry in data:
            row = f"{entry['timestamp']},{entry['temperature']},{entry['humidity']},{entry['pressure']},{entry['gas']}\n"
            yield row

    return app.response_class(generate(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=sensor_data.csv"})

if __name__ == "__main__":
    # Run Flask on all network interfaces, port 5000
    app.run(host="0.0.0.0", port=8001, debug=True)
