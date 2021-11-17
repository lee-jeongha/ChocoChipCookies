var cnt = 0; // 제대로 답변을 마친 질문 개수
if (!("webkitSpeechRecognition" in window)) {
       alert("이 브라우저는 stt를 지원하지 않습니다.")
}

window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

let recognition = new SpeechRecognition();
recognition.interimResults = false; //임시 결과 반환 여부를 제어. true는 실시간으로 인식된 결과 확인, false는 최종 결과만 확인
recognition.lang = 'ko-KR'; //언어는 한국어
recognition.continuous = true; //기본값은 false. true로 설정할 경우, 각각 인식된 문장을 하나로 합쳐주며, 중간에 쉬어도 stop되지 않음.
recognition.onresult = function(e) {
    var texts = Array.from(e.results).map(results => results[0].transcript).join("");
    var p = document.getElementById("temp");
    p.textContent = p.textContent + texts;
}

window.onload = function() { // 페이지가 로드되면 첫번째 질문과 답변 bubble을 보여준다. 첫번째 질문 TTS를 실행한다.
    start();
}

var speak_q = function(event) {
    speak(event.target.innerText);
};

function start() {
    var q = document.getElementById("q0");
    q.classList.replace('behind', 'show');
    q.addEventListener('click', speak_q);
    setTimeout(function() {document.getElementById("a0").classList.replace('behind', 'show');}, 1000);
    var btn = document.getElementById("yes0");
    btn.disabled = false;
    btn.style.cursor = "pointer";
    btn = document.getElementById("no0");
    btn.disabled = false;
    btn.style.cursor = "pointer";
}

function yes_click(i) { // 답변하기. 현재 답변 bubble을 녹음 중인 div로 바꿔주고 STT를 시작한다.
    var s = String(i);
    document.getElementById("ab" + s).classList.replace('show', 'behind');
    var btn = document.getElementById("yes" + s);
    btn.disabled = true;
    btn.style.cursor = "default";
    btn = document.getElementById("no" + s);
    btn.disabled = true;
    btn.style.cursor = "default";
    document.getElementById("al" + s).classList.replace('behind', 'show');
    btn = document.getElementById("done" + s);
    btn.disabled = false;
    btn.style.cursor = "pointer";
    btn.style.height = "50px";
    // STT 시작
    recognition.start();
}

function no_click(i, num) { // 넘어가기. 현재 답변 bubble을 숨기고 다음 질문과 답변 bubble을 보여준다.
    var s = String(i);
    var q = document.getElementById("q" + s);
    var a = document.getElementById("a" + s);
    q.removeEventListener('click', speak_q);
    a.classList.replace('show', 'behind');
    setTimeout(function() {a.style.minHeight = "0px";}, 1000);
    var btn = document.getElementById("yes" + s);
    btn.disabled = true;
    btn.style.cursor = "default";
    btn = document.getElementById("no" + s);
    btn.disabled = true;
    btn.style.cursor = "default";
    if (i < num - 1) {
        s = String(i + 1);
        var q = document.getElementById('q' + s);
        q.classList.replace('behind', 'show');
        q.addEventListener('click', speak_q);
        setTimeout(function() {document.getElementById("a" + s).classList.replace('behind', 'show');}, 1000);
        var btn = document.getElementById("yes" + s);
        btn.disabled = false;
        btn.style.cursor = "pointer";
        btn = document.getElementById("no" + s);
        btn.disabled = false;
        btn.style.cursor = "pointer";
    } else { // 마지막 질문이면 다운로드 버튼 보여주기
        document.getElementById("last").classList.replace("behind", "show");
        var btn = document.getElementById("down");
        btn.classList.replace("behind", "show");
        btn.disabled = false;
        btn.style.cursor = "pointer";
    }
}

function done_click(i, num) { // 답변 끝내기. 현재 div를 숨기고 STT로 변환된 텍스트를 출력한다. 무사히 답변을 마친 질문이라고 표시해준다. 다음 질문과 답변 bubble을 보여준다.
    recognition.abort();
    var p = document.getElementById("temp");
    cnt += 1;
    var s = String(i);
    document.getElementById("al" + s).classList.replace('show', 'behind');
    var btn = document.getElementById("done" + s);
    btn.disabled = true;
    btn.style.cursor = "default";
    btn.style.height = "0px";
    var ac = document.getElementById("ac" + s);
    ac.textContent = p.textContent;
    p.textContent = "";
    ac.classList.replace('behind', 'show');
    var ctn = document.getElementById("a" + s);
    ctn.style.minHeight = "0px";

    var q = document.getElementById('q' + s);
    var a = document.getElementById('ac' + s);
    var result = document.getElementById('result');
    result.innerText += ("\n\n　　　──────────\n\n\n　　　(Q)　" + q.innerText + "\n\n　　　(A)　" + a.innerText + "\n");
    q.removeEventListener('click', speak_q);
    if (i < num - 1) {
        s = String(i + 1);
        var q = document.getElementById('q' + s);
        q.classList.replace('behind', 'show');
        q.addEventListener('click', speak_q);
        setTimeout(function() {document.getElementById("a" + s).classList.replace('behind', 'show');}, 1000);
        var btn = document.getElementById("yes" + s);
        btn.disabled = false;
        btn.style.cursor = "pointer";
        btn = document.getElementById("no" + s);
        btn.disabled = false;
        btn.style.cursor = "pointer";
    } else {
        document.getElementById("last").classList.replace("behind", "show");
        var btn = document.getElementById("down");
        btn.classList.replace("behind", "show");
        btn.disabled = false;
        btn.style.cursor = "pointer";
    }
}

function download() {
    var element = document.createElement('a');
    var textInput = document.getElementById('result');
    element.setAttribute('href', 'data:text/plain;charset=utf-8, ' + encodeURIComponent(textInput.innerText));
    element.setAttribute('download', '모의면접결과.txt');
    document.body.appendChild(element);
    element.click();
    //document.body.removeChild(element);
}

