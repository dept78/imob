
let speed = 0;
let rpm = 1000;
let fuelLevel = 75;
let gear = 'N';
let maxGears = 5;
let engineRunning = false;
let accelerating = false;
let braking = false;
let throttle = 0;
let brakePressure = 0;
let engineTemp = 90; 
let coolantTemp = 85; 
let batteryVoltage = 12.5; 
let oilPressure = 30; 
let fuelConsumption = 8.5; 

const maxSpeedByGear = { 
    N: 0, 
    1: 30, 
    2: 60, 
    3: 100, 
    4: 140, 
    5: 180 
};

const speedDisplay = document.getElementById('speed');
const rpmDisplay = document.getElementById('rpm');
const gearDisplay = document.getElementById('gearDisplay');
const fuelDisplay = document.getElementById('fuelDisplay');
const engineTempDisplay = document.getElementById('engineTempDisplay');
const coolantTempDisplay = document.getElementById('coolantTempDisplay');
const batteryVoltageDisplay = document.getElementById('batteryVoltageDisplay');
const oilPressureDisplay = document.getElementById('oilPressureDisplay');
const fuelConsumptionDisplays = document.getElementById('fuelConsumptionDisplays');
const throttleDisplay = document.getElementById('throttleDisplay');
const brakePressureDisplay = document.getElementById('brakePressureDisplay');
const engineSound = document.getElementById('engineSound');
const brakeSound = document.getElementById('brakeSound');

function updateDisplay() {
    speedDisplay.innerText = speed.toFixed(1);
    rpmDisplay.innerText = rpm;
    gearDisplay.innerText = `Gear: ${gear}`;
    fuelDisplay.innerText = `Fuel: ${fuelLevel.toFixed(1)}%`;
    engineTempDisplay.innerText = `Engine Temp: ${engineTemp.toFixed(1)}°C`;
    coolantTempDisplay.innerText = `Coolant Temp: ${coolantTemp.toFixed(1)}°C`;
    batteryVoltageDisplay.innerText = `Battery: ${batteryVoltage.toFixed(1)}V`;
    oilPressureDisplay.innerText = `Oil Pressure: ${oilPressure.toFixed(1)} PSI`;
    fuelConsumptionDisplays.innerText = `Fuel Consumption: ${fuelConsumption.toFixed(1)} L/100km`;
    throttleDisplay.innerText = `Throttle: ${throttle}%`;
    brakePressureDisplay.innerText = `Brake Pressure: ${brakePressure}%`;

    engineSound.playbackRate = Math.min(Math.max(rpm / 1000, 0.5), 3);
    engineSound.volume = speed > 0 ? 1 : 0; 
}

function getPredictions() {
    const inputData = {
        speed: speed,
        throttle: throttle,
        brake_pressure: brakePressure,
        rpm: rpm,
        fuel_level: fuelLevel,
        fuel_consumption:fuelConsumption,
        gear: gear,
        engine_temp: engineTemp,
        battery_voltage: batteryVoltage,
        oil_pressure: oilPressure
    };

    fetch("/predict", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify(inputData),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('fuelConsumptionDisplay').innerText = `Remaining Range: ${data.fuel_efficiency.toFixed(0)}`;
        document.getElementById('maintenanceWarnings').innerText = `Maintenance Alerts: ${data.maintenance_warnings.join(', ') || 'No issues'}`;
        if (data.maintenance_warnings.includes("High engine temperature!")) {
          var s=new Audio()
        }
        document.getElementById('brakeStatus').innerText = `Brake Status: ${data.brake_status}`;
        document.getElementById('gearSuggestion').innerText = `Gear Suggestion: ${data.gear_suggestion}`;
        document.getElementById('driverBehavior').innerText = `Driver Behavior: ${data.driver_behavior}`;
        console.log(data.fuel_efficiency);
    })
    .catch((error) => {
        console.error('Error:', error);
        
    });
    function getCookie(name) {
      let cookieValue = null;
      if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
          const cookie = cookies[i].trim();
          if (cookie.substring(0, name.length + 1) === (name + '=')) {
            cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
            break;
          }
        }
      }
      return cookieValue;
    }
  
}

if(speed>0){
setInterval(() => {
    getPredictions();
    
}, 500);  // Send updated data every 5 seconds
}
// Start accelerating
function startAccelerating() {
    if (!accelerating && gear !== 'N') {
        accelerating = true;
        engineRunning = true;
        engineSound.play();

        const accelerateInterval = setInterval(() => {
            if (accelerating && speed < maxSpeedByGear[gear]) {
                throttle = Math.min(throttle + 5, 100); // Increase throttle up to 100%
                brakePressure = 0; // No brake pressure during acceleration

                speed += 1; 
                rpm = Math.min(rpm + 150, 6000);

                // Adjust other values during acceleration
                engineTemp = Math.min(engineTemp + 0.1, 110); // Engine temperature increases
                coolantTemp = Math.min(coolantTemp + 0.05, 100); // Coolant temperature rises slowly
                oilPressure = Math.min(oilPressure + 0.1, 50); // Oil pressure increases
                fuelConsumption = Math.min(fuelConsumption + 0.1, 15); // Fuel consumption rises
                fuelLevel = Math.max(fuelLevel - 0.01, 0); // Fuel level decreases
                updateDisplay();
                
                getPredictions();
            } else {
                clearInterval(accelerateInterval);
            }
        }, 100);
    }
}

if(!accelerating && gear !== 'N'){
    batteryVoltage = Math.max(batteryVoltage - 0.01, 0)
}
// Start braking
function startBraking() {
    if (!braking && speed > 0) {
        braking = true;
        brakeSound.play();

        const brakeInterval = setInterval(() => {
            if (braking && speed > 0) {
                brakePressure = Math.min(brakePressure + 10, 100); // Increase brake pressure
                throttle = 0; // Throttle is 0 during braking

                speed -= 2;
                rpm = Math.max(rpm - 500, 1000);
                // Adjust other values during braking
                engineTemp = Math.max(engineTemp - 0.05, 90); // Engine temperature drops slightly
                coolantTemp = Math.max(coolantTemp - 0.02, 85); // Coolant temperature stabilizes
                oilPressure = Math.max(oilPressure - 0.1, 30); // Oil pressure decreases
                fuelConsumption = Math.max(fuelConsumption - 0.05, 8); // Fuel consumption decreases

                updateDisplay();
                
                getPredictions();
            } else {
                clearInterval(brakeInterval);
                brakeSound.pause();
            }
        }, 100);
    } else if (speed < 0) {
        speed = 0;
    }
}

// Gear up function
function gearUp() {
    if (gear === 'N') {
        gear = 1; // Switch from Neutral to 1st gear
    } else if (gear < maxGears) {
        gear++;
    }
    updateDisplay();
    
    getPredictions();
}

// Gear down function
function gearDown() {
    if (gear > 1) {
        gear--;
    } else {
        gear = 'N'; // Switch to Neutral
        engineSound.pause();
        engineRunning = false;
    }
    updateDisplay();
    
    getPredictions();
}

// Pedal and gear buttons
const accelerateBtn = document.getElementById('accelerateBtn');
const brakeBtn = document.getElementById('brakeBtn');
const gearUpBtn = document.getElementById('gearUpBtn');
const gearDownBtn = document.getElementById('gearDownBtn');

// Event listeners for pedals and gears
accelerateBtn.addEventListener('mousedown', startAccelerating);
accelerateBtn.addEventListener('mouseup', () => accelerating = false);
brakeBtn.addEventListener('mousedown', startBraking);
brakeBtn.addEventListener('mouseup', () => braking = false);
gearUpBtn.addEventListener('click', gearUp);
gearDownBtn.addEventListener('click', gearDown);

// Reduce speed over time when not accelerating
setInterval(() => {
    if (!accelerating && speed > 0) {
        speed -= 0.5; // Slow down gradually
        rpm = Math.max(rpm - 500, 1000);
        brakePressure = Math.max(brakePressure-10,0)
        // Adjust other values during deceleration
        engineTemp = Math.max(engineTemp - 0.02, 90); // Gradual cool-down
        coolantTemp = Math.max(coolantTemp - 0.01, 85);
        fuelConsumption = Math.max(fuelConsumption - 0.01, 8);
        updateDisplay();
        
        getPredictions();
    }
}, 200);
