# DataAnalyze Project
## 고용률 분석 및 데이터 전처리 프로젝트

---

## 📋 **프로젝트 개요**

이 프로젝트는 한국의 지역별 고용률과 E9 체류자수(외국인 노동자) 간의 관계를 분석하고, 데이터 전처리 파이프라인을 구축하는 것을 목표로 합니다.

### **주요 목표**
- 📊 **데이터 전처리**: 원시 데이터를 분석 가능한 형태로 변환
- 🔍 **데이터 검증**: 데이터 품질 및 일관성 검증
- 📈 **고용률 분석**: E9 체류자수와 고용률의 관계 분석
- 🎯 **정책 제언**: 분석 결과를 바탕으로 한 정책적 함의 도출

---

## 🏗️ **프로젝트 구조**

```
DataAnalyze/
├── 📁 src/                          # 소스 코드
│   ├── 📁 processing/               # 데이터 전처리 모듈들
│   │   ├── area_master.py          # 지역 마스터 데이터 처리
│   │   ├── area_processor.py       # 지역별 데이터 처리
│   │   ├── config.py               # 설정 파일
│   │   ├── data_merger.py          # 데이터 병합
│   │   ├── data_processor.py       # 데이터 전처리
│   │   ├── e9_processor.py         # E9 체류자 데이터 처리
│   │   ├── employment_processor.py # 고용 데이터 처리
│   │   ├── industry_processor.py   # 산업별 데이터 처리
│   │   ├── main_preprocessor.py    # 메인 전처리 파이프라인
│   │   ├── data_validation_analysis.py    # 데이터 검증 분석
│   │   └── detailed_data_validation.py    # 상세 데이터 검증
│   ├── 📁 analyze/                 # 분석 모듈들
│   │   ├── employment_analysis.py  # 고용률 분석 (기본 다중회귀)
│   │   ├── panel_analysis.py       # 패널분석 (내생성 통제)
│   │   ├── enhanced_panel_analysis.py  # 향상된 패널분석 (상호작용 포함)
│   │   └── analysis_results.md     # 분석 결과 보고서 (한글)
│   ├── 📁 utils/                   # 유틸리티 모듈들
│   │   └── test_data.py           # 테스트 데이터 생성
│   ├── __init__.py
│   └── main.py                     # 메인 실행 파일
├── 📁 data/                         # 데이터 파일들
│   ├── 📥 raw/                     # 원시 데이터
│   └── 📤 processed/               # 전처리된 데이터
├── 📁 outputs/                      # 분석 결과물
│   ├── enhanced_interaction_analysis.png    # 상호작용 효과 분석
│   ├── panel_analysis_results.png          # 패널분석 결과
│   └── panel_residual_analysis.png         # 패널분석 잔차분석
├── 📁 analysis_docs/                # 분석 문서
├── 📁 code_docs/                    # 코드 문서
├── 📁 logs/                         # 로그 파일
├── 📁 venv/                         # 가상환경
├── requirements.txt                 # Python 패키지 의존성
└── README.md                        # 프로젝트 설명서
```

---

## 🚀 **빠른 시작**

### **1. 환경 설정**
```bash
# 저장소 클론
git clone <repository-url>
cd DataAnalyze

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### **2. 분석 실행**

#### **메인 메뉴 실행**
```bash
cd src
python3 main.py
```

#### **전체 기간 통합 E9-고용률 분석 (신규 추가) ⭐**
```bash
# 2019-2023년 전체 기간 고용률 데이터 보유 지자체 대상 종합 분석
python3 comprehensive_employment_analysis.py  # 전체 기간 통합 분석 (분리하지 않음)
```

#### **기존 구조적 관계 분석**
```bash
# 패널분석 (권장 - 내생성 통제)
python3 analyze/panel_analysis.py

# 향상된 패널분석 (상호작용 효과)
python3 analyze/enhanced_panel_analysis.py

# 기본 고용률 분석 (참고용)
python3 analyze/employment_analysis.py
```

---

## 📊 **주요 분석 결과**

### 🆕 **전체 기간 통합 E9-고용률 분석 결과 (2019-2023)**

#### **🎯 핵심 발견사항**
- **전체 기간 통합**: 2019-2023년 완전 데이터 보유 지자체 대상 종합 분석
- **현실적 E9 효과**: 0.06-0.07%p의 미미하지만 일관된 양의 효과 확인
- **지역 이질성**: 농어촌 상위(파랑) vs 도시/공업 하위(주황)의 뚜렷한 구분
- **정책 방향**: 단순하고 일관된 정책이 복잡한 차별화보다 효율적

#### **📈 주요 통계**
- **분석 대상**: 153개 지자체 (2019-2023년 전체 기간 고용률 데이터 보유)
- **분석 기간**: 2019-2023년 완전 통합 (분리하지 않음)
- **E9 효과**: 기본 모형 0.0598, 향상 모형 0.0674
- **설명력**: Within R² 0.240 (높은 신뢰성)

### 📊 **구조적 관계 분석 결과 (패널분석, 2019-2023)**

### **🎯 핵심 발견사항**
- **내생성 통제**: 패널분석을 통해 관측되지 않은 지역 특성 통제
- **Two-way Fixed Effects**: 개체(지역) + 시간(연도) 고정효과로 엄격한 인과추론
- **상호작용 효과**: E9 외국인의 효과가 산업구조에 따라 차별적으로 나타남

### **📈 패널분석 모델 성능**
- **Basic Panel**: Within R² = 16.65% (개체 고정효과)
- **Two-way FE**: Within R² = 6.05% (개체+시간 고정효과)
- **Enhanced Model**: 상호작용 효과 포함으로 더 정교한 분석

### **🔍 Hausman 검정 결과**
- **권장 모형**: Random Effects (p = 1.000)
- **모형 선택**: 통계적 검정을 통한 객관적 모형 선택

---

---

## 📁 **생성된 시각화 파일**

### 🆕 **전체 기간 통합 E9-고용률 분석 시각화**
- **2019_2023_종합_E9고용률_분석결과.png**: 전체 기간 통합 종합 분석 결과 (9개 차트)
- **2019_2023_패널분석_계수결과.csv**: 기본/향상 모형 계수 비교 테이블
- **2019_2023_지역별_고용률_순위.csv**: 지역별 고용률 순위 (상위 파랑, 하위 주황)
- **2019_2023_종합분석_요약.txt**: 종합 분석 결과 및 정책 시사점

### 📊 **기존 구조적 분석 시각화**

### **1. panel_analysis_results.png**
- **의미**: 패널분석 모형별 계수 비교 및 모형 적합도
- **내용**: FE, RE, Two-way FE 모형의 계수 추정치와 신뢰구간
- **핵심**: 내생성 통제 후 변수별 효과의 변화

### **2. panel_residual_analysis.png**
- **의미**: 패널분석 잔차 진단
- **내용**: 적합값 vs 잔차, 잔차 분포 히스토그램
- **핵심**: 모형의 타당성 및 가정 충족 여부 확인

### **3. enhanced_interaction_analysis.png**
- **의미**: 상호작용 효과 분석 및 조건부 한계효과
- **내용**: 산업구조에 따른 E9 효과의 차별적 영향
- **핵심**: Johnson-Neyman 구간을 통한 정책적 함의 도출

---

## 🔧 **기술 스택**

### **프로그래밍 언어**
- **Python 3.10+**: 메인 프로그래밍 언어

### **데이터 처리**
- **pandas**: 데이터 조작 및 분석
- **numpy**: 수치 계산
- **scipy**: 통계 분석

### **시각화**
- **matplotlib**: 기본 그래프 생성
- **seaborn**: 통계적 시각화

### **패널분석**
- **linearmodels**: 패널데이터 분석 (Fixed Effects, Random Effects)
- **statsmodels**: 통계 모델링 및 검정

### **머신러닝**
- **scikit-learn**: 기본 회귀분석 및 모델링

---

## 📚 **사용법 가이드**

### **1. 데이터 전처리**
```python
from src.processing.data_processor import DataProcessor
from src.processing.config import Config

config = Config()
processor = DataProcessor(config)
processor.process_all_data()
```

### **2. 패널분석 (권장)**
```python
from src.analyze.panel_analysis import PanelAnalyzer

analyzer = PanelAnalyzer()
df_yearly = analyzer.load_and_preprocess_data()
panel_df, dependent_var, fe_vars, re_vars = analyzer.prepare_panel_data()
fe_result = analyzer.run_fixed_effects_model(panel_df, dependent_var, fe_vars)
analyzer.create_visualizations()
```

### **3. 향상된 패널분석 (상호작용 포함)**
```python
from src.analyze.enhanced_panel_analysis import EnhancedPanelAnalyzer

analyzer = EnhancedPanelAnalyzer()
df_yearly = analyzer.load_and_preprocess_data()
panel_df = analyzer.prepare_panel_data()
result = analyzer.run_enhanced_panel_model()
analyzer.calculate_marginal_effects()
analyzer.create_interaction_visualizations()
```

### **4. 데이터 검증**
```python
from src.processing.data_validation_analysis import generate_validation_report

generate_validation_report()
```

---

## 📋 **데이터 소스**

### **원시 데이터**
- **E9 체류자 데이터**: 시군구별 E9 체류자수 (2014~2023)
- **산업별 고용 데이터**: 시군구별 제조업/서비스업 종사자수
- **고용률 데이터**: 시군구별 연령대별 고용률
- **지역 정보**: 시도/시군구별 면적 및 행정구역 정보

### **전처리된 데이터**
- **최종 통합 데이터**: 모든 원시 데이터를 통합한 분석용 데이터셋
- **형식**: CSV (UTF-8, CP949 인코딩 지원)
- **기간**: 2021~2023년 (COVID-19 기간 제외)

---

## 🔍 **분석 방법론**

### **1. 데이터 전처리**
- COVID-19 기간 제외 (2021년 이후)
- 반기별 → 연도별 집계
- 밀도 기반 변수 생성 (절대값 → 단위면적당 비율)
- 로그 변환 (0값 처리 및 분포 정규화)

### **2. 분석 기법**
- **패널분석**: Fixed Effects, Random Effects, Two-way Fixed Effects
- **내생성 통제**: 관측되지 않은 지역 특성 통제를 통한 인과추론
- **상호작용 분석**: 조건부 한계효과 및 Johnson-Neyman 구간 분석
- **모형 선택**: Hausman 검정을 통한 객관적 모형 선택
- **통계적 검정**: 클러스터 표준오차를 통한 강건한 추론

### **3. 변수 정의**
- **E9_밀도**: E9 체류자수 / 면적
- **제조업_집중도**: 제조업 종사자수 / 총 종사자수
- **종사자_밀도**: 총 종사자수 / 면적
- **수도권**: 서울/경기/인천 여부 (이진변수)

---

## 📊 **정책적 함의**

### **1. E9 체류자 정책**
- **밀도 관리**: 지역별 E9 체류자 밀도 조절 필요
- **과밀화 방지**: 특정 지역에 집중되지 않도록 분산 배치

### **2. 지역 개발 정책**
- **비수도권 우수성**: 비수도권의 고용률 우수성 확인
- **지역 균형**: 수도권 집중 현상 완화 필요

### **3. 산업 정책**
- **산업 구조**: 제조업/서비스업 비중보다는 지역 특성이 중요
- **밀도 관리**: 종사자 밀도 조절을 통한 고용률 개선

---

## 🚧 **제한사항 및 주의사항**

### **데이터 제한사항**
- **기존 분석**: COVID-19 기간(2019-2020) 데이터 품질 문제로 제외
- **신규 분석**: 단기 기간(2019-2020) 분석으로 장기 트렌드 파악 한계
- **지역**: 일부 도서 지역 데이터 누락 가능성
- **산업**: 제조업/서비스업 이분법적 구분

### **분석 제한사항**
- **시간 범위**: 3년간(2021-2023)의 짧은 기간으로 장기 효과 추정 한계
- **변수 제약**: 관측 가능한 변수로만 구성되어 일부 설명되지 않은 이질성 존재
- **외부 충격**: COVID-19와 같은 구조적 변화의 영향 완전히 배제 어려움

---

## 🔮 **향후 개발 계획**

### **단기 계획 (1-3개월)**
- [x] COVID-19 초기 충격 분석 완료 ⭐
- [ ] 2019-2023년 통합 분석 (기존+신규 결합)
- [ ] 머신러닝 모델 적용
- [ ] 대시보드 개발

### **중기 계획 (3-6개월)**
- [ ] 실시간 데이터 업데이트 시스템
- [ ] 정책 효과 분석 모듈
- [ ] API 서비스 개발

### **장기 계획 (6개월 이상)**
- [ ] 다른 국가 데이터와의 비교 분석
- [ ] 예측 모델 개발
- [ ] 학술 논문 발표

---

## 🤝 **기여 방법**

### **버그 리포트**
- GitHub Issues를 통해 버그 신고
- 상세한 재현 단계와 환경 정보 포함

### **기능 제안**
- 새로운 분석 방법이나 시각화 기법 제안
- 코드 예시와 함께 제안

### **코드 기여**
- Fork 후 Pull Request 방식
- 코드 스타일 가이드 준수

---

## 📄 **라이선스**

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 👥 **팀 정보**

### **개발자**
- **주요 개발자**: [이름]
- **분석가**: [이름]
- **데이터 엔지니어**: [이름]

### **연락처**
- **이메일**: [email@example.com]
- **GitHub**: [GitHub 프로필]
- **블로그**: [블로그 URL]

---

## 📚 **참고 자료**

### **학술 논문**
- [관련 연구 1]
- [관련 연구 2]
- [관련 연구 3]

### **데이터 소스**
- [통계청]
- [고용노동부]
- [지방자치단체]

### **기술 문서**
- [pandas 공식 문서]
- [matplotlib 튜토리얼]
- [scikit-learn 가이드]

---

## 📝 **변경 이력**

### **v1.1.0 (2025-09-25) ⭐ 신규**
- 🎯 **전체 기간 통합 E9-고용률 분석 추가**
- 📊 2019-2023년 전체 기간 고용률 데이터 보유 지자체 대상 종합 분석
- 📈 완전 통합 접근으로 분석 신뢰성 확보 (R² 0.240)
- 📋 05번 분석 보고서 신규 작성 (04번과 상호보완)
- 🎨 상위(파랑)/하위(주황) 차별적 색상 시각화로 정책적 실용성 제고

### **v1.0.0 (2025-08-31)**
- 🎉 초기 버전 릴리스
- 📊 고용률 분석 모듈 완성
- 🔄 데이터 전처리 파이프라인 구축
- 📈 시각화 및 결과 보고서 생성

---

*마지막 업데이트: 2025년 9월 25일*
*프로젝트 버전: v1.1.0*

**🆕 주요 업데이트 내용:**
- 전체 기간 통합 E9-고용률 분석 신규 추가
- 2019-2023년 전체 기간 고용률 데이터 보유 지자체 대상 완전 통합 분석
- 04번 보고서와 상호보완적 이중 분석 체계 구축 (포괄성 ↔ 정밀성)
- 상위(농어촌, 파랑) vs 하위(도시/공업, 주황) 차별적 시각화 구현
