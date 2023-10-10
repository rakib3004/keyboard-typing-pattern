let pressed = [];
let released = [];
let tempStore = {};
let startOn = Date.now();
let escapable = ["control", "shift", "alt", "escape", "capslock", "numlock"];
let sentence = "abcdefghijklmnopqrstuvwxyz ";
let attributes = getAttributes();


function clearKeyStore() {
    pressed = [];
    released = [];
    tempStore = {};
}

function properChar(char) {
    if (char == " ") return "space";
    if (char == ",") return "comma";
    if (char == ".") return "period";
    if (char == ";") return "semicln";
    if (char == ":") return "colon";
    if (char == "\n") return "enter";
    // if (char == ) return "Target";

    return char;
}

function getKey(event) {
    return event.key.toLowerCase();
}

async function keyUpFunc(event) {
    let key = getKey(event);
    if (escapable.includes(key)) return;
    if (key == "backspace") {
        released.pop();
        return;
    }
    if (key != "enter") {
    released.push({ key, time: Date.now() });
    // console.log(`Released ${key} on ${released[released.length - 1].time - startOn} ms`);
    if (released.length > 1) {
        let UUtime = released[released.length - 1].time - released[released.length - 2].time;
        let prevKey = released[released.length - 2].key;
        let currKey = released[released.length - 1].key;
        // console.log(`${prevKey} and ${currKey} up up time ${UUtime} ms`);
    }
}

else {
    await process();
}
}

function keyDownFunc(event) {
    let key = getKey(event);
    if (escapable.includes(key)) return;
    if (key == "backspace") {
        pressed.pop();
        return;
    }
    if (pressed.length == 0) {
        startOn = Date.now();
    }
    pressed.push({ key, time: Date.now() });
    // console.log(`Pressed ${key} on ${pressed[pressed.length - 1].time - startOn} ms`);
    if (pressed.length > 1) {
        let DDtime = pressed[pressed.length - 1].time - pressed[pressed.length - 2].time;
        let prevKey = pressed[pressed.length - 2].key;
        let currKey = pressed[pressed.length - 1].key;
        // console.log(`${prevKey} and ${currKey} down down time ${DDtime} ms`);
    }
    if (released.length > 0) {
        let UDtime = pressed[pressed.length - 1].time - released[released.length - 1].time;
        let prevKey = released[released.length - 1].key;
        let currKey = pressed[pressed.length - 1].key;
        // console.log(`${prevKey} and ${currKey} up down time ${UDtime} ms`);
    }
}

function getAttributes() {
    let headers = new Set(['user'])
    for (let i = 0; i < sentence.length; i++) {
        for (let j = 0; j < sentence.length; j++) {
            currentChar = properChar(sentence.charAt(i));
            nextChar = properChar(sentence.charAt(j));
            headers.add(`H.${currentChar}`);
            headers.add(`DD.${currentChar}.${nextChar}`);
            headers.add(`UD.${currentChar}.${nextChar}`);
        }
    }
    return Array.from(headers);
}

function updateIntermediateArray(key1, key2, value, prefix) {
    // tempStore contains frequency, finalValue, sum
    let keyName;
    if (prefix === "H") {
        keyName = `H.${key1}`;
    } else {
        keyName = `${prefix}.${key1}.${key2}`;
    }

    if (tempStore[keyName]) {
        // update operation
        tempStore[keyName].frequency++;
        tempStore[keyName].sum += value;
        tempStore[keyName].finalValue = tempStore[keyName].sum / tempStore[keyName].frequency;
    } else {
        tempStore[keyName] = { frequency: 1, finalValue: value, sum: value };
    }
}

function getPattern(email) {
    // need change here
    let processed = [];
    let processed2 = [email];
    for (let index in pressed) {
        if (pressed[index].key != released[index].key) {
            console.log(`Noted (${pressed[index].key}-${released[index].key})`)
        }
        if (!sentence.includes(pressed[index].key)) {
            throw new Error("Unrecognized character" + pressed[index].key);
        }
        let pressTime = (pressed[index].time - startOn) / 1000.0;
        let releaseTime = (released[index].time - startOn) / 1000.0;
        processed.push({ key: properChar(pressed[index].key), pressTime, releaseTime });
    }

    for (let i = 1; i < processed.length; i++) {
        current = processed[i - 1];
        next = processed[i];
        updateIntermediateArray(current.key, next.key, current.releaseTime - current.pressTime, "H");
        updateIntermediateArray(current.key, next.key, next.pressTime - current.pressTime, "DD");
        updateIntermediateArray(current.key, next.key, next.pressTime - current.releaseTime, "UD");
        if (i == processed.length - 1) {
            updateIntermediateArray(next.key, next.key, next.releaseTime - next.pressTime, "H");
        }
    }
    console.log('-----temp----store--------',tempStore);

    let values = [];
    let request_values = new Map();
        for (attrName of attributes) {
            if(attrName==='user'){
                continue;
            }
       else if (tempStore[attrName]) {
            values.push(tempStore[attrName].finalValue);
            request_values.set(attrName, tempStore[attrName].finalValue); 
        }
        
        else { values.push(0); 
            request_values.set(attrName, 0);
        }
    }

    let request_values_object = {};
for (let [key, value] of request_values) {
    request_values_object[key] = value;
}
request_values_object['Unnamed: 1486']=0;
    console.log('expected output -----------', request_values_object)
    return request_values_object;
}

async function process() {
    let email = document.getElementById("recover-text-input").value;
    let typing_pattern_data = getPattern(email);
    
    await verifiedUser(typing_pattern_data);

}


async function verifiedUser(typing_pattern_data){

    const userElement = document.getElementById('user');

    try {
        const response = await fetch('http://127.0.0.1:5000/pattern', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(typing_pattern_data)
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();


        const jsonData = JSON.stringify(data);
        const parsedData = JSON.parse(jsonData);
        const resultValue = parsedData.result * 100;
        if(resultValue>50){
            userElement.textContent = "Geniune";
        }
        else{

            userElement.textContent = "Imposter"
        }

    } catch (error) {
        console.error('Error:', error);
    }

}

async function saveToFile(data) {
    let existingData;
    try {
        [tempFile] = await window.showOpenFilePicker();
        let file = await tempFile.getFile();
        let contents = await file.text();
        existingData = contents.endsWith("\n") ? contents : contents + "\n";
    } catch (e) {
        // user cancelled file select dialog
        existingData = attributes.join(",") + "\n";
    }
    download(existingData + data + "\n");
}

function download(content) {
    let blob = new Blob([content]);
    let blobUrl = URL.createObjectURL(blob);
    // Create a link element
    const link = document.createElement("a");

    // Set link's href to point to the Blob URL
    link.href = blobUrl;
    link.download = "testpattern.csv";

    // Append link to the body
    document.body.appendChild(link);

    // Dispatch click event on the link
    // This is necessary as link.click() does not work on the latest firefox
    link.dispatchEvent(
        new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            view: window
        })
    );

    // Remove link from body
    document.body.removeChild(link);
}

async function register(email, password) {
    let body = { email, password };
    let resp = await fetch("http://localhost:5000/register", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    let resp_json = await resp.json();
    alert(resp_json.status);
    return resp_json;
}

async function login(email, password) {
    let body = { email, password };
    let resp = await fetch("http://localhost:5000/login", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    let resp_json = await resp.json();
    if (resp_json.status === "success") {
        // go to logged in page
    }
    alert(resp_json.status);
    return resp_json;
}

async function recovery() {
    let email = document.getElementById("recover-input-email").value;
    let pattern = getPattern(email);
    let body = { email, pattern };
    let resp = await fetch("http://localhost:5000/recovery", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    let resp_json = await resp.json();
    if (resp_json.status === "success") {
        // go to logged in page
    }
    alert(resp_json.status);
    return resp_json;
}
