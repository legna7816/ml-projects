# NLP 감성분석 (Sentiment Analysis)

영화 리뷰 텍스트를 긍정/부정으로 분류하는 감성분석 프로젝트.
텍스트를 숫자로 표현하는 방법을 BoW → TF-IDF → 워드 임베딩 순으로 직접 구현하며 비교했다.

## 사용 기술
- Python, scikit-learn, gensim
- TF-IDF, Logistic Regression, Naive Bayes
- GloVe 워드 임베딩

## 데이터
- Stanford IMDB Movie Review (긍정/부정 각 25,000개)

## 진행 과정
1. **텍스트 벡터화**: BoW(CountVectorizer)와 TF-IDF로 텍스트를 숫자 벡터로 변환
2. **감성분석 모델**: TF-IDF + Logistic Regression / Naive Bayes로 긍정·부정 분류
3. **워드 임베딩**: GloVe 임베딩으로 단어의 의미적 관계 탐색

## 결과
- TF-IDF + Logistic Regression: 정확도 **약 86%** (샘플 10,000개 기준)
- 데이터 양 증가 시 Logistic Regression 성능 향상 확인 (5,000개 84% → 10,000개 86%)
- bigram(ngram_range) 적용이 감성분석 정확도에 유리함을 확인

## 배운 점 (핵심 인사이트)
- **데이터 누수 방지**: 새 데이터 예측 시 `fit_transform`이 아닌 `transform`만 사용해야 하는 이유를 이해
- **정적 임베딩의 한계**: GloVe 임베딩이 `happy`-`sad`(반의어)를 `happy`-`joyful`(유의어)보다 더 가깝게 판단 → 같은 문맥에서 쓰이는 반의어를 구분하지 못하는 한계 발견
- 이 한계가 문맥 기반 임베딩(BERT 등 Transformer)이 등장한 배경임을 이해
