# AI System Project 2026

온디바이스 AI 반도체 시스템 설계 과정의 AI 파트에서 실습한 코드를 정리한 저장소입니다.
Python 기초, 머신러닝 기초, CNN, Keras 실습, 라즈베리파이 추론 순서로 진행했습니다.

> 이 과정을 바탕으로 만든 팀 프로젝트: [WAKEY 미션 알람시계](https://github.com/gurwn031107-web/wakey-mission-alarm-clock)

## 구성

| 폴더 | 분야 | 핵심 내용 | 도구 |
|---|---|---|---|
| [`2026_Python_Basic`](2026_Python_Basic) | Python 기초 | 자료형, 입출력, 클래스·객체, 컨테이너, 함수, 조건문, 리스트 메서드, 모듈·패키지 | Python, VS Code |
| [`2026_ML_DL_Basic`](2026_ML_DL_Basic) | 머신러닝 기초 | 퍼셉트론, 가설 최적화, 선형·로지스틱 회귀, Softmax 분류, MNIST | NumPy, TensorFlow/Keras |
| [`2026_CNN`](2026_CNN) | 합성곱 신경망 | Conv 1층 / 3층 MNIST 분류, ReLU, Dropout 적용 | Keras |
| [`2026_Keras_LAB`](2026_Keras_LAB) | 실습 과제 | CSV 데이터 회귀·분류, 논리 게이트 SLP/MLP, 균열 이미지 CNN | Keras, Pandas, OpenCV |
| [`2026_Edge_AI`](2026_Edge_AI) | 엣지 AI | 라즈베리파이 카메라 + TFLite 실시간 추론 | LiteRT(TFLite), OpenCV |

## 2026_Python_Basic

| 파일 | 내용 |
|---|---|
| `Py_Lab_for_VS_Code.py` | 1~12장 실습: 출력, 변수, 자료형, 클래스와 객체, 컨테이너(list/tuple/dict/set), built-in 함수, 함수 정의, if 문, list 메서드, 마트 셀프 계산기 |
| `my_module.py`, `your_module.py`, `my_package/` | 모듈 import와 패키지 구조 실습 |

## 2026_ML_DL_Basic

| 파일 | 내용 |
|---|---|
| `EX_01_Perceptron` | 퍼셉트론 연산 (가중합 + 계단 함수) |
| `EX_02_Optimize_Hypothesis` | 경사하강법으로 가설 최적화 |
| `EX_03_Blood_Pressure_Linear_Regression` | 혈압 예측 선형 회귀 |
| `EX_04_N-variable_M-output_LR_Keras` | 다변수, 다출력 선형 회귀 (Keras) |
| `EX_05_Blood_Pressure_Logistic_Regression` | 로지스틱 회귀 이진 분류 |
| `EX_06_Blood_Pressure_Softmax_Classification` | Softmax 다중 분류 |
| `EX_07_MNIST_Classification` | 완전연결층 MNIST 손글씨 분류 |

## 2026_CNN

| 파일 | 내용 |
|---|---|
| `EX_01_1_Conv_MNIST_Classification` | Conv 1층 CNN |
| `EX_02_3_Conv_MNIST_Classification` | Conv 3층 CNN |
| `EX_03_MNIST_ReLU_Dropout` | ReLU, Dropout으로 과적합 줄이기 |

## 2026_Keras_LAB

| 파일 | 내용 |
|---|---|
| `LAB_04_linear_regression.py` | CSV 데이터 선형 회귀 (MSE + SGD) |
| `LAB_05_binary_classification_blood.py` | 나이·BMI로 고혈압 여부 이진 분류 |
| `LAB_06_multinomial_classification_hitters.py` | MLP 32-16-3으로 연봉 상/중/하 다중 분류 |
| `LAB_07_0_gate_classification_SLP.ipynb` | 단층 퍼셉트론으로 논리 게이트 분류 + 결정 경계 |
| `LAB_07_1_gate_classification_MLP.ipynb` | 다층 퍼셉트론으로 XOR 등 비선형 게이트 분류 |
| `LAB_09_cnn_crack_detection.py` | 3 Conv + ReLU + Dropout CNN으로 표면 균열 판별 |

## 2026_Edge_AI

| 파일 | 내용 |
|---|---|
| `mnist_tflite_inference.py` | 라즈베리파이 카메라 영상에서 손글씨 숫자를 TFLite 모델로 실시간 인식 |

## ▶ 실행

- `.ipynb` / `LAB_*.py`: Google Colab에서 실행. 데이터 파일은 각 파일 상단 안내대로 `내 드라이브`에 넣습니다.
- `mnist_tflite_inference.py`: 라즈베리파이에서 `ai-edge-litert`, `opencv-python` 설치 후 같은 폴더에 `mnist_baseline_model.tflite`를 두고 실행합니다.

> 데이터셋, 모델 파일(`.tflite`, `.keras`), 정답 풀이는 저장소에서 제외했습니다.
