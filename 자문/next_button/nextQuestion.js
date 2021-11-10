//DOM
const NextButton = document.querySelector(".next-button");

//function
function print_name(list) {
    // 결과를 표시할 element
    const resultElement = document.getElementById('result');
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
        alert("이 브라우저는 음성 합성을 지원하지 않습니다.")
        return
    }

    window.speechSynthesis.cancel() // 현재 읽고있다면 초기화

    const prop = opt_prop || {}

    const speechMsg = new SpeechSynthesisUtterance()
    speechMsg.rate = prop.rate || 1 // 속도: 0.1 ~ 10
    speechMsg.pitch = prop.pitch || 1 // 음높이: 0 ~ 2
    speechMsg.lang = prop.lang || "ko-KR"
    speechMsg.text = text

    // SpeechSynthesisUtterance에 저장된 내용을 바탕으로 음성합성 실행
    window.speechSynthesis.speak(speechMsg)
}