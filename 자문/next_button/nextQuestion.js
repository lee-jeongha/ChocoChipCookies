//DOM
const endButton = document.querySelector(".end-interview");

//function
function print_name(list) {
    // 결과를 표시할 element
    const resultElement = document.getElementById('question');
    const countElement = document.getElementById('count');

    // 현재 화면에 표시된 값
    let number = countElement.innerText;

    // int형으로 바꾸기
    number = parseInt(number);

    // 결과 출력 : resultElement.innerText = list;
    if (number<list.length-1){
        //질문 글 출력
        resultElement.innerHTML = list[number];

        //tts로 질문 읽어주기 (밑의 speak()함수 참고)
        speak(list[number], { rate: 0.9, pitch: 0.8, lang: "ko-KR" }); //속도는 0.9 음높이는 0.5 언어는 한국어

        //숫자 세기(이 숫자는 화면에는 보이지 않음)
        countElement.innerText = ++number;

        //3초 뒤 질문 글 없애기
        setTimeout(function(){
            resultElement.innerHTML = ""; }, 3000);
    }
}

//Web Speech API의 speechSynthesis를 이용하여 tts 구현하기
function speak(text, opt_prop) {
    if (typeof SpeechSynthesisUtterance === "undefined" || typeof window.speechSynthesis === "undefined") {
        alert("이 브라우저는 tts를 지원하지 않습니다.")
        return
    }

    window.speechSynthesis.cancel() // 현재 문장을 읽고 있다면 초기화

    const prop = opt_prop || {}

    const speechMsg = new SpeechSynthesisUtterance()
    speechMsg.rate = prop.rate || 1 // 속도: 0.1 ~ 10
    speechMsg.pitch = prop.pitch || 1 // 음높이: 0 ~ 2
    speechMsg.lang = prop.lang || "ko-KR"
    speechMsg.text = text

    // SpeechSynthesisUtterance에 저장된 내용을 바탕으로 음성합성 실행
    window.speechSynthesis.speak(speechMsg)
}

function listen(){

    if (!("webkitSpeechRecognition" in window)) {
        alert("이 브라우저는 stt를 지원하지 않습니다.")
        return
    }

    window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    let recognition = new SpeechRecognition();
    recognition.interimResults = true; //임시 결과 반환 여부를 제어. true는 실시간으로 인식된 결과 확인, false는 최종 결과만 확인
    recognition.lang = 'ko-KR'; //언어는 한국어
    recognition.continuous = true; //기본값은 false. true로 설정할 경우, 각각 인식된 문장을 하나로 합쳐주며, 중간에 쉬어도 stop되지 않음.

    //새로운 문단을 추가하는 함수
    let makeNewTextContent = function() {
        p = document.createElement('p'); //'p'라는 이름의 Element를 생성한다.
        document.querySelector('.answer').appendChild(p); //answer 자리에 붙인다.
    };

    let p = null;

    recognition.start(); //음성 인식(녹음) 시작
    recognition.onstart = function() {
        makeNewTextContent(); // 음성 인식 시작시마다 새로운 문단을 추가한다.
    };

    /*//자동으로 종료될 때마다 다른 문장을 또 인식하기 위해 onend를 사용해 다시 시작할 수 있게 설정
    recognition.onend = function() {
        recognition.start();
    };*/

    recognition.onresult = function(e) {
        let texts = Array.from(e.results)
            .map(results => results[0].transcript).join(""); //우리가 원하는 인식된 문장은 results의 첫번째 Alternative값의 transcript에 담겨있다.

        p.textContent = texts;
    };

    //"면접 끝내기" 버튼 누르면 인식 종료
    /*endButton.onclick = function() {
        recognition.abort();
        console.log('Speech recognition aborted.');
    };*/

    endButton.addEventListener('click', function(){
        recognition.abort();
    });
}