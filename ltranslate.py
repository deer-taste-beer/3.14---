import pyperclip

while True:
    # 사용자로부터 여러 줄의 문장 입력 받기
    input_sentences = []
    print("여러 줄의 문장을 입력하세요. 입력이 끝나면 엔터를 누르세요.(90 입력시 종료)")

    # 엔터를 입력할 때까지 계속 입력 받기
    while True:
        line = input()
        if not line:  # 사용자가 엔터를 누르면 종료
            break
        input_sentences.append(line)

    # 입력 받은 여러 줄의 문장을 한 줄로 결합하고 좌우 공백 정리
    processed_text = ' '.join(input_sentences).strip()

    # "" 문자를 제거
    processed_text = processed_text.replace('', '')
    # 만약에 입력이 "90"이면 루프를 종료
    if processed_text == "90":
            90
            print("프로그램을 종료합니다.")
            break
    # 처리된 문장 출력
    print("처리된 문장:", processed_text)

    # 클립보드에 복사
    pyperclip.copy(processed_text)
    print("클립보드로 복사되었습니다.")

    
