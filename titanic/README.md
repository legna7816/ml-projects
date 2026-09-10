# 타이타닉 생존자 예측 (Titanic Survival Prediction)

정형 데이터를 이용한 머신러닝 분류 문제. 
탑승자 특성(나이, 성별, 객실등급 등)을 보고 생존 여부를 예측하는 프로젝트.

## 사용 기술
- Python, pandas, scikit-learn
- EDA(Exploratory Data Analysis), 결측치 처리, 전처리
- Logistic Regression, Decision Tree, Random Forest
- 교차검증(Cross Validation)

## 데이터
- Kaggle Titanic Dataset (891명, 12개 컬럼)
- 타겟: survived (0=사망, 1=생존)

## 프로젝트 흐름

### 1. EDA (탐색적 데이터 분석)
- `sex`(성별): 여성 생존율 74% vs 남성 19% → **성별이 생존에 큰 영향**
- `pclass`(객실등급): 1등급 63% vs 3등급 24% → **등급도 생존에 영향**

### 2. 전처리
- **결측치 처리**
  - age: 중앙값(median)으로 채우기 (평균보다 이상치에 덜 민감)
  - embarked: 최빈값으로 채우기 (결측치 2개만)
- **범주형 변수 인코딩**
  - sex: male/female → 0/1
  - embarked: one-hot encoding (다중공선성 방지로 drop_first=True)
- **Feature Engineering**
  - family_size = sibsp + parch (동승 가족 수)
  - is_alone = 1 if family_size==0 else 0

### 3. 모델 학습 & 평가
- **데이터 분리**: 80/20 (학습/평가)
- **모델 3가지 비교**
  - Logistic Regression: 정확도 79% (안정적)
  - Decision Tree: 정확도 93% (단회, 과적합 의심)
  - Random Forest: 정확도 81% (중간)

### 4. 교차검증으로 신뢰성 확인
| 모델 | 5-fold CV 평균 |
|---|---|
| Logistic | 0.973 |
| DecisionTree | 0.953 |
| RandomForest | 0.814 |

**발견**: 단회 train_test_split에서 DecisionTree가 0.93으로 제일 높았지만, 
교차검증 결과는 Logistic이 0.973으로 가장 안정적. 
"1번 결과가 좋다" ≠ "진짜 좋은 모델" (우연일 수 있음)

## 핵심 인사이트

### 다중공선성(Multicollinearity) 발견
```
Feature Importance Top 3:
- sex: 0.271
- fare: 0.263
- age: 0.249
- pclass: (상위 3에 없음!)
```

**왜 pclass가 안 나타났을까?**
- EDA에서 pclass가 생존에 영향을 주는 걸 확인했는데 (1등급 63% vs 3등급 24%)
- 모델의 feature importance에서는 fare가 더 높게 나옴
- 이유: **pclass와 fare가 강하게 연관** (1등급 탑승자 = 고가 티켓)
- 모델 입장에선 둘 중 하나(fare)에 중요도를 몰아주고, 나머지(pclass)는 낮게 평가

**교훈**: 
- Feature importance는 "독립적인 영향력"이 아니라 "모델이 가장 먼저 본 패턴"
- 다중공선성이 있으면 변수 선택에 주의 필요
- 모델 결과와 EDA 결과가 다를 수 있음 (의문을 가져야 함)

### 데이터 크기의 중요성
- 데이터가 충분하지 않아도 ML은 작동하지만, **신뢰도가 떨어짐**
- 이 프로젝트는 891명으로 작은 편이라 교차검증의 분산이 컸음
- 타이타닉은 유명 데이터라 괜찮지만, 실무에선 "최소 몇 천 개?" 기준을 항상 고민해야 함

## 배운 것 (다음 프로젝트에 적용)
1. **fit/predict 패턴** — 모든 scikit-learn 모델이 동일 (이 패턴은 NLP/이미지에도 똑같이 적용됨)
2. **train/test 분리의 필수성** — 학습 데이터로 평가하면 의미 없음
3. **교차검증** — 1회 결과만 믿지 말 것, 여러 번 나눠서 평균으로 판단
4. **EDA 후 모델 → 다시 EDA로 돌아와 해석** — 순환 과정이 중요
