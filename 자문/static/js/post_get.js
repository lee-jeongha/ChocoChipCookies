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
        resultElement.innerHTML = list[number];
        countElement.innerText = ++number;
    }
}

//events
//NextButton.addEventListener("click", get_name);