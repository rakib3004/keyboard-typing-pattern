let pressed = [];
let released = [];
let startOn = Date.now();
let escapable = ["Control", "Shift", "Alt", "Escape", "CapsLock", "NumLock"];
let sentence = ".tie5Roanl\n";


function clearKeyStore() {
    pressed = [];
    released = [];
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
    return event.key;
}

function keyUpFunc(event) {
    let key = getKey(event);
    if (escapable.includes(key)) return;
    if (key == "Backspace") {
        released.pop();
        return;
    }
    released.push({ key, time: Date.now() });
    console.log(`Released ${key} on ${released[released.length - 1].time - startOn} ms`);
    if (released.length > 1) {
        let UUtime = released[released.length - 1].time - released[released.length - 2].time;
        let prevKey = released[released.length - 2].key;
        let currKey = released[released.length - 1].key;
        console.log(`${prevKey} and ${currKey} up up time ${UUtime} ms`);
    }
    if (key == "Enter") {
        process();
    }
}

function keyDownFunc(event) {
    let key = getKey(event);
    if (escapable.includes(key)) return;
    if (key == "Backspace") {
        pressed.pop();
        return;
    }
    if (pressed.length == 0) {
        startOn = Date.now();
    }
    pressed.push({ key, time: Date.now() });
    console.log(`Pressed ${key} on ${pressed[pressed.length - 1].time - startOn} ms`);
    if (pressed.length > 1) {
        let DDtime = pressed[pressed.length - 1].time - pressed[pressed.length - 2].time;
        let prevKey = pressed[pressed.length - 2].key;
        let currKey = pressed[pressed.length - 1].key;
        console.log(`${prevKey} and ${currKey} down down time ${DDtime} ms`);
    }
    if (released.length > 0) {
        let UDtime = pressed[pressed.length - 1].time - released[released.length - 1].time;
        let prevKey = released[released.length - 1].key;
        let currKey = pressed[pressed.length - 1].key;
        console.log(`${prevKey} and ${currKey} up down time ${UDtime} ms`);
    }
}

async function process() {
    let password = document.getElementById("recover-text-input").value;
    let confirm = document.getElementById("recover-text-input").value;
    if (password < 3) {
        alert("Password is too short");
        return;
    }
    if (password != confirm) {
        alert("Confirm password not match");
        return;
    }
    if (pressed.length != released.length) {
        alert("Something not right");
        return;
    }
    let email = document.getElementById("recover-input-email").value;
    // await register(email,password);
    let processed = [];
    let processed2 = [email];
    for (let index in pressed) {
        if (pressed[index].key != released[index].key) {
            console.log(`Noted (${pressed[index].key}-${released[index].key})`)
        }
        let pressTime = (pressed[index].time - startOn) / 1000.0;
        let releaseTime = (released[index].time - startOn) / 1000.0;
        processed.push({ key: pressed[index].key, pressTime, releaseTime });
    }
    console.log(processed);
    for (let i = 1; i < processed.length; i++) {
        current = processed[i - 1];
        next = processed[i];
        processed2.push(current.releaseTime - current.pressTime);
        processed2.push(next.pressTime - current.pressTime);
        processed2.push(next.pressTime - current.releaseTime);
        if (i == processed.length - 1) {
            processed2.push(next.releaseTime - next.pressTime);
        }
    }
    // await saveToFile(processed2.join(","));
    await recover(email, processed2);
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
        let headers = ["user"];

        for (let i = 1; i < sentence.length; i++) {
            currentChar = properChar(sentence.charAt(i - 1));
            nextChar = properChar(sentence.charAt(i));
            headers.push(`H.${currentChar}`);
            headers.push(`DD.${currentChar}.${nextChar}`);
            headers.push(`UD.${currentChar}.${nextChar}`);
            if (i == sentence.length - 1) {
                headers.push(`H.${nextChar}`);
            }
        }
        existingData = headers.join(",") + "\n";
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
    console.log(resp_json);
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
    console.log(resp_json);
    if (resp_json.status === "success") {
        // go to logged in page
    }
    alert(resp_json.status);
    return resp_json;
}
async function recover(email, pattern) {
    let body = { email, pattern };
    let resp = await fetch("http://localhost:5000/recover", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    let resp_json = await resp.json();
    console.log(resp_json);

    alert(resp_json.status);
    return resp_json;
}