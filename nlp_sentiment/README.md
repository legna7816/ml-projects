# NLP 감성분석 (Sentiment Analysis)

영화 리뷰 텍스트를 긍정/부정으로 분류하는 감성분석 프로젝트.
**텍스트를 숫자로 표현하는 방법의 발전 과정**을 BoW → TF-IDF → 워드 임베딩 → BERT(문맥 임베딩) → BERT 파인튜닝 순으로 직접 구현하며 비교했다.

## 사용 기술
- Python, scikit-learn, gensim, Hugging Face `transformers`, PyTorch
- TF-IDF, Logistic Regression, Naive Bayes
- GloVe 워드 임베딩
- BERT (bert-base-uncased) — 추론 및 파인튜닝

## 데이터
- Stanford IMDB Movie Review (긍정/부정 각 25,000개)

## 파일 구성 (순서대로 보면 전체 스토리가 이어짐)

| 파일 | 내용 |
|---|---|
| `nlp_step01_tokenizer.ipynb` | 토큰화, BoW, TF-IDF 원리 실습 |
| `nlp_step02_movie.ipynb` | 소규모 리뷰 데이터로 감성분석 첫 실습 |
| `nlp_step03_IMDB.ipynb` | 실제 IMDB 데이터(5,000~10,000개)로 TF-IDF 감성분석 |
| `nlp_step04_wordEmbedding.ipynb` | GloVe 워드 임베딩, 의미 연산 및 한계 탐색 |
| `nlp_step05_BERT.ipynb` | BERT 문맥 임베딩 실습 (추론만) |
| `nlp_step06_BERT_finetuning.ipynb` | BERT 파인튜닝으로 감성분석 직접 학습 |

## 진행 과정 & 핵심 실험

### 1. 텍스트 벡터화 (step01)
- **BoW**: 단어 등장 횟수로 벡터화. 문장 간 의미 차이를 반영하지 못함
- **TF-IDF**: 흔한 단어(the, was)는 가중치를 낮추고, 구별력 있는 단어(delicious, terrible)는 높게 부여
- **stop_words**: 의미 없는 흔한 단어 자동 제거 (단, 'not' 같은 부정어까지 같이 제거되는 부작용 존재)

### 2. 실전 감성분석 (step02~03)
- TF-IDF + Logistic Regression / Naive Bayes로 IMDB 리뷰 분류
- **결과**: 샘플 5,000개 84.5% → 10,000개 86.4% (Logistic은 데이터 늘수록 성능 향상, NaiveBayes는 거의 정체)
- **ngram_range=(1,2)**: bigram 포함 시 unigram만 쓸 때보다 정확도 소폭 상승 (84.1%→84.5%)
- **데이터 누수 방지**: 새 데이터 예측 시 `fit_transform`이 아닌 `transform`만 사용해야 하는 이유 확인

### 3. 워드 임베딩 (step04)
- GloVe(정적 임베딩)로 단어 간 의미 관계 탐색
- **성공 사례**: `king - man + woman = queen`, `paris - france + italy = rome` — 벡터 공간이 관계를 학습함을 확인
- **한계 발견**: `happy`-`sad`(반의어) 유사도 0.68 > `happy`-`joyful`(유의어) 유사도 0.53 → **반의어를 구분하지 못함**
  - 이유: good/bad, happy/sad는 같은 문맥에서 함께 쓰이기 때문에 정적 임베딩이 이를 "비슷한 단어"로 학습함

### 4. BERT 문맥 임베딩 (step05)
GloVe의 한계를 문맥으로 극복하는지 직접 검증:
- **다의어 구분 성공**: `bank`(은행 vs 강둑) 문맥별 벡터 유사도 0.34 → 문맥에 따라 다른 벡터 생성 확인
- **반의어 구분 개선**: 단어 벡터 직접 비교 시 `happy`-`joyful` 0.757 > `happy`-`sad` 0.691로 **GloVe와 순서가 역전** (문제 해결)
- **문장 평균(pooling)의 함정**: 문장 전체를 평균 내면 반의어 차이가 희석되어 여전히 구분 실패 → pooling 전략이 성능에 큰 영향을 준다는 것을 확인
- **Subword 토큰화**: `unhappiness` → `un`+`##ha`+`##pp`+`##iness`처럼 조각으로 쪼개져, 사전에 없는 단어도 처리 가능

### 5. BERT 파인튜닝 (step06)
사전학습된 BERT에 분류 레이어를 얹고 IMDB 데이터로 직접 학습:
- **환경**: GPU 필요 (추론은 CPU 가능하나 파인튜닝은 GPU로 전환)
- **결과**: 2,000개 샘플, 2 epoch 기준 정확도 86.5% (TF-IDF 10,000개 결과와 유사한 수준을 훨씬 적은 데이터로 달성)
- **실패 사례 발견**: "the movie was not bad" → 부정으로 오분류. 파인튜닝해도 미묘한 이중부정 뉘앙스는 놓칠 수 있음을 확인
- **실험 설계 교훈**: 여러 조건(epoch, 데이터 크기)을 비교할 때 모델을 매번 사전학습 상태에서 새로 불러오지 않으면, 이전 학습 효과가 누적되어 잘못된 결론에 이를 수 있음을 직접 경험 (공정한 비교의 중요성)
- **작은 검증 세트의 함정**: 데이터를 500개로 줄였을 때 오히려 정확도가 90%로 상승 — 검증 세트가 100개로 작아지면 결과가 통계적으로 불안정해질 수 있음

## 핵심 인사이트 요약

| 방법 | 의미 파악 | 문맥 구분 | 데이터 효율 |
|---|---|---|---|
| BoW/TF-IDF | ❌ | ❌ | 데이터 많이 필요 |
| GloVe(정적 임베딩) | ✅ 유의어 | ❌ 반의어 혼동 | 사전학습된 것 재사용 |
| BERT(문맥 임베딩) | ✅ | ✅ 문맥별 벡터 다름 | 파인튜닝 시 적은 데이터로도 효과적 |

**텍스트 표현 방법이 발전해온 이유가 이 프로젝트 안에서 실험으로 재현됨**: 단어 등장 횟수만 세던 것에서 → 의미를 벡터로 담고 → 문맥까지 반영하는 방향으로.

## 배운 점 (다음 프로젝트에 적용)
- fit/predict 패턴은 scikit-learn이든 Hugging Face Trainer든 동일하게 적용됨
- 벡터화/임베딩 방식이 결국 검색 기반 시스템(RAG)의 핵심 원리와 동일함 → 다음 프로젝트(RAG)로 자연스럽게 연결
- 실험 비교 시 통제 변인 관리(모델 초기화, random_state, 검증 세트 크기)가 결과 신뢰성에 결정적임
