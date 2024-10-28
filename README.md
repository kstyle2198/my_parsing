# parser_v02.py 사용방법

1. Main Functions
   - main_filepath_extractor : 최상위 폴더 경로를 입력받아, 폴더 트리를 재귀적으로 탐색하면서 모든 pdf 파일의 Full paths를 리스트에 저장
   - main_parser : pdf파일의 full path 등을 입력받아 파싱 수행
     (텍스트파싱, 테이블파싱, PDF를 PNG 이미지로 저장, 커스텀 메타데이터 추출-폴더명, 이미지PDF는 OCR 적용 등)
