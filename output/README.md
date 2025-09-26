# 고용률-E9체류자 분석 결과 (2019-2023)

## 데이터 소스
- **주데이터**: `data/processed/최종_통합데이터_수정_utf-8.csv`
- **경계데이터**: southkorea-maps GitHub repository (KOSTAT 2013 기준)
- **좌표계**: EPSG:4326 (WGS84)

## 분석 방법론
- **패널기간**: 2019-2023 (5년)
- **분석단위**: 시군구 (균형패널)
- **색상분류**: Quantiles (k=5) - 대비 강화
- **회귀모형**: 고용률 = β × log(1 + E9체류자수) + 고정효과

## 산출물 목록

### 📊 데이터
- `panel/panel_balanced_2019_2023.csv` - 균형패널 데이터
- `panel/sgg_list_153.csv` - 분석대상 시군구 목록
- `summary/avg_5yr_by_sgg.csv` - 시군구별 5년 평균

### 📈 모델
- `models/fe_regression.txt` - 고정효과 패널회귀
- `models/pooled_ols.txt` - 통합 OLS (비교용)

### 🗺️ 지도
- `maps/employment_rate_5yr_avg.png` - 고용률 5년 평균
- `maps/e9_5yr_avg.png` - E-9체류자 5년 평균
- `maps/employment_rate_2019.png` - 고용률 2019
- `maps/employment_rate_2023.png` - 고용률 2023
- `maps/e9_2019.png` - E-9체류자 2019
- `maps/e9_2023.png` - E-9체류자 2023

### 🔍 디버깅
- `debug/unmatched_keys.csv` - 매칭 실패 시군구
- `cache/sgg_boundary.geojson` - 경계 캐시파일

## 주요 가정사항
1. 반기별 데이터는 연도별로 집계 (E9: 합계, 고용률: 평균)
2. 시도+시군구 조합으로 고유 식별코드 생성
3. 5년 연속 데이터 보유 시군구만 분석 포함
4. 결측치 제거 후 분석 진행

## 분석 실행
```bash
python comprehensive_employment_analysis.py
```

---
*생성일: 2025년 | 분석도구: Python, GeoPandas, LinearModels*
